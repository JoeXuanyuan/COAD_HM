#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 01:44:35 2022

@author: xuanyuanqiao
"""

import os 
import random
import numpy as np
import pandas as pd
import cv2
import torch
import math
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from torchvision import transforms
import utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def Get_split_loader(split_dataset, training = False, weighted = True):

    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    if training:
        if weighted:
            weights = Make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL_Training, **kwargs)	
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL_Training, **kwargs)
    else:
        loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL_Testing, **kwargs)	
    return loader



##############################################################################


def Make_weights_for_balanced_classes_split(dataset):
    
    N = float(len(dataset))    
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.Getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)


def collate_MIL_Training(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_MIL_Testing(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    coords = torch.LongTensor([item[2] for item in batch])
    return [img, label, coords]



def GetTiles(csvFile, maxTileNum, label_dict, test = False, seed = 23, filterPatients = []):

    np.random.seed(seed)
    data = pd.read_csv(csvFile)
    
    if not len(filterPatients) == 0:
        patientsUnique = filterPatients
    else:
        patientsUnique = list(set(data['PATIENT']))        
    
    tilesPathList = []
    yTrueList = []
    yTrueLabelList = []
    patinetList = []
    
    for index, patientID in enumerate(tqdm(patientsUnique)):
        selectedData = data.loc[data['PATIENT'] == patientID]
        selectedData.reset_index(inplace = True)
        tempTiles = []
        for item in range(len(selectedData)):
            tempTiles.extend([os.path.join(selectedData['SlideAdr'][item], i) for i in os.listdir(selectedData['SlideAdr'][item])])
        if len(tempTiles) > maxTileNum:
            random.shuffle(tempTiles)
            tempTiles = np.random.choice(tempTiles, maxTileNum, replace=False)
        for tile in tempTiles:
            tilesPathList.append(tile)
            yTrueList.append(utils.get_value_from_key(label_dict, selectedData['label'][0]))
            yTrueLabelList.append(selectedData['label'][0])
            patinetList.append(str(patientID))
                
    df = pd.DataFrame(list(zip(patinetList, tilesPathList, yTrueList, yTrueLabelList)), columns =['PATIENT', 'TilePath', 'yTrue', 'yTrueLabel'])     
    df_temp = df.dropna()
    
    if test:
        dfFromDict = df_temp
    else:            
        tags = list(df_temp['yTrue'].unique())
        tagsLength = []
        dfs = {}
        for tag in tags:
            temp = df_temp.loc[df_temp['yTrue'] == tag]
            temp = temp.sample(frac=1).reset_index(drop=True)
            dfs[tag] = temp 
            tagsLength.append(len(df_temp.loc[df_temp['yTrue'] == tag]))
        
        minSize = np.min(tagsLength)
        keys = list(dfs.keys())
        frames = []
        for key in keys:
            temp_len = len(dfs[key])
            diff_len = temp_len - minSize
            drop_indices = np.random.choice(dfs[key].index, diff_len, replace = False)
            frames.append(dfs[key].drop(drop_indices))
            
        dfFromDict = pd.concat(frames)
                    
    return dfFromDict

###############################################################################   

class DatasetLoader_Classic(torch.utils.data.Dataset):

    def __init__(self, imgs, labels, transform = None, target_patch_size = -1):
        self.labels = labels
        self.imgs = imgs
        self.target_patch_size = target_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        X = Image.open(self.imgs[index])
        y = self.labels[index]
        if self.target_patch_size is  not None:
            X = X.resize((self.target_patch_size, self.target_patch_size))
            X = np.array(X)
        if self.transform is not None:
            X = self.transform(X)
        return X, y

###############################################################################

def LoadTrainTestFromFolders(trainPath, testPath):
    
    pathContent = os.listdir(testPath)
    pathContent = [os.path.join(testPath , i) for i in pathContent]
    
    test_x = []
    test_y = []
    
    for path in pathContent:
        if path.split('\\')[-1] == 'MSIH':
            y = 1
        else:
            y = 0
        tiles = os.listdir(path)
        tiles = [os.path.join(path , i) for i in tiles]
        test_x = test_x + tiles
        test_y = test_y + [y]* len(tiles)
    
    pathContent = os.listdir(trainPath)
    pathContent = [os.path.join(trainPath , i) for i in pathContent]
    
    train_x = []
    train_y = []
    
    for path in pathContent:
        if path.split('\\')[-1] == 'MSIH':
            y = 1
        else:
            y = 0
        tiles = os.listdir(path)
        tiles = [os.path.join(path , i) for i in tiles]
        train_x = train_x + tiles
        train_y = train_y + [y]* len(tiles)
        
    return train_x, train_y, test_x, test_y

###############################################################################
    

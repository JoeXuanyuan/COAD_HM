#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 02:02:32 2022

@author: xuanyuanqiao

"""

from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import torch
import os
import random
from sklearn import preprocessing
from tqdm import tqdm
import argparse
from efficientnet_pytorch import EfficientNet

from utils.data_utils import DatasetLoader_Classic, GetTiles
from utils.core_utils import Train_model_Classic, Validate_model_Classic
from eval.eval import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
import utils.utils as utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################

def generate_ds(imagesPath, annotPath, outputPath):
    
    labelList_return = []
    patientList_return = []
    
    patientList = []
    labelList = []
    slideList = []
    slideAdr = []
    
    table = pd.read_csv(annotPath)
    caseList= list(table["case_id"])
    patients = list(set(caseList))
    
    #labelsList = list(set(list(table["label"])))

    imageNames = os.listdir(imagesPath)
    imageNames = [os.path.join(imagesPath, i) for i in imageNames]

    
    for patientID in tqdm(patients):
        indicies = [i for i, n in enumerate(caseList) if n == patientID]
        matchedSlides = [list(table['slide_id'])[i] for i in indicies] 
        matchedLabels = [list(table['label'])[i] for i in indicies] 

        for slide in matchedSlides:
            
            temp = [i for i in imageNames if slide in i]
            slideName = temp[0]

            patientList.append(patientID)
            slideList.append(slideName.split('/')[-1])
            slideAdr.append(slideName) 
            labelList.append(matchedLabels[0])

            if not patientID in patientList_return:
                patientList_return.append(patientID)
                labelList_return.append(matchedLabels[0])   
        
    data = pd.DataFrame()
    data["PATIENT"] = patientList
    data["FILENAME"] = slideList
    data["SlideAdr"] = slideAdr
    data["label"] = labelList
    
    data.to_csv(os.path.join(outputPath,'Clean_data.csv'), index = False)
    
    return patientList_return, labelList_return, os.path.join(outputPath,'Clean_data.csv')                 


###############################################################################

def Initialize_model(num_classes, use_pretrained = True):
  
    model_ft = None
    input_size = 0

    model_ft = EfficientNet.from_pretrained('efficientnet-b6')
    num_ftrs = model_ft._fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 512

    return model_ft, input_size
  
###############################################################################

def Classic_Training(args):
    
    random.seed(args.seed)
    
    folder = args.output_dir
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
        
    args.split_dir = os.path.join(args.output_dir,"SPLITS")
    if not os.path.isdir(args.split_dir):
        os.makedirs(args.split_dir)

    args.result_dir = os.path.join(args.output_dir,"RESULT")
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
    
    reportFile  = open(os.path.join(args.output_dir,'Report.txt'), 'a', encoding="utf-8")
    reportFile.write('-' * 30 + '\n')
    reportFile.write(str(args))
    reportFile.write('-' * 30 + '\n')
    
    patientsList, labelsList, cleanCSV = generate_ds(imagesPath = args.datadir_train, annotPath = args.csvFile, outputPath=args.output_dir)
    
    
    le = preprocessing.LabelEncoder()
    labelsList = le.fit_transform(labelsList)
    
    args.n_classes = len(set(labelsList))
    args.label_dict = dict(zip(le.classes_, range(len(le.classes_))))
    ## {'Hypermutated': 0, 'Non-hypermutated': 1}
    
    utils.Summarize(args, list(labelsList), reportFile)
    
    ###########################################
    #######    Start Cross Validation   #######
    ###########################################
    
    print('It is a ' + str(args.k) + ' fold cross validation training !')
    
    patientID = np.array(patientsList)
    labels = np.array(labelsList)
    
    folds = args.k
    kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
    kf.get_n_splits(patientID, labels)
    
    foldcounter = 1
    
    for train_index, test_index in kf.split(patientID, labels):
        
        testPatients = patientID[test_index]
        trainPatients = patientID[train_index]
        
        print('Generate new tiles...\n')

        print('For train set ... \n')
        train_data = GetTiles(csvFile = cleanCSV, maxTileNum = args.maxTileNum, label_dict = args.label_dict, test = False, filterPatients = trainPatients)

        print('For validation set ...\n')
        val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.15))                
        val_x = list(val_data['TilePath']) 
        val_y = list(val_data['yTrue'])                  

        train_data = train_data[~train_data['TilePath'].isin(val_x)]
        train_x = list(train_data['TilePath'])
        train_y = list(train_data['yTrue']) 
        
        print('For test set ...\n')
        test_data = GetTiles(csvFile = cleanCSV, label_dict = args.label_dict, maxTileNum = args.maxTileNum, test = True, filterPatients = testPatients)
        test_x = list(test_data['TilePath'])
        test_y = list(test_data['yTrue']) 
        
        
        test_data.to_csv(os.path.join(args.split_dir, 'TestSplit_' + str(foldcounter) + '.csv'), index = False)
        train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit_' + str(foldcounter) + '.csv'), index = False)
        val_data.to_csv(os.path.join(args.split_dir, 'ValSplit_' + str(foldcounter) + '.csv'), index = False)                       

        print('-' * 30)
        print("K FOLD validation setp => {}".format(foldcounter))  
        print('-' * 30)
        
        model, input_size = Initialize_model(args.n_classes, use_pretrained = True)

        if args.use_ckpt:
            print("Use pretrained checkpoint ... ")
            model_path = args.ckpt_path
            try:
                model.load_state_dict(torch.load(model_path))
            except:
                model = torch.load(model_path)

        model.to(device)
        

        train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
        trainGenerator = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    
        val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
        valGenerator = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size, shuffle = True)
      
        test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
        testGenerator = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size, shuffle = False)
        
        noOfLayers = 0
        for name, child in model.named_children():
            noOfLayers += 1
        
        cut = int (args.freeze_Ratio * noOfLayers)
        
        ct = 0
        for name, child in model.named_children():
            ct += 1
            if ct < cut:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
                
        temp = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(temp, lr=args.lr, weight_decay=args.reg)

        criterion = nn.CrossEntropyLoss()
        
        print('\n')
        print('Start training ...')
        
        model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_Classic(model = model, trainLoaders = trainGenerator, valLoaders = valGenerator, criterion = criterion, optimizer = optimizer, args = args, fold = str(foldcounter))            
        print('-' * 30)
                                    
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'finalModelFold' + str(foldcounter)))
        history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)), 
                                  columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        
        history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FOLD_' + str(foldcounter) + '.csv'), index = False)
        print('\nSTART EVALUATION ON TEST DATA SET ...', end = ' ')
        
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'bestModelFold' + str(foldcounter))))
        probsList  = Validate_model_Classic(model = model, dataloaders = testGenerator)

        probs = {}
        for key in list(args.label_dict.keys()):
            probs[key] = []
            for item in probsList:
                probs[key].append(item[utils.get_value_from_key(args.label_dict, key)])
    
        probs = pd.DataFrame.from_dict(probs)
        
        testResults = pd.concat([test_data, probs], axis = 1)                    
        testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_BASED_FOLD_' + str(foldcounter) + '.csv')
        testResults.to_csv(testResultsPath, index = False)
        
        CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = foldcounter , reportFile = reportFile)
        reportFile.write('-' * 30 + '\n')                
        foldcounter +=  1 

              
    patientScoreFiles = []
    tileScoreFiles = []

    for i in range(args.k):
        patientScoreFiles.append('TEST_RESULT_PATIENT_BASED_FOLD_' + str(i+1) + '.csv')
        tileScoreFiles.append('TEST_RESULT_TILE_BASED_FOLD_' + str(i+1) + '.csv')  
        
    CalculateTotalROC(resultsPath = args.result_dir, results = patientScoreFiles, target_labelDict =  args.label_dict, reportFile = reportFile)
    reportFile.write('-' * 30 + '\n')
    MergeResultCSV(args.result_dir, tileScoreFiles)
    reportFile.close()


##############################################################################
    
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--datadir_train', type=str, default=None, 
                    help='WSI directory, normalized and filtered tumor tiles')
parser.add_argument('--csvFile', type=str, default=None, 
                    help='annotation file')
parser.add_argument('--output_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--use_ckpt', action='store_true', default=False,
                    help='whether to use already trained model checkpoint')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='trained model checkpoint path')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=3,
                    help='number of folds (default: 10)')
parser.add_argument('--maxTileNum', type=int, default=200,
                    help='maximum number of images in a WSI bag to train (default: 500)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--early_stopping', action='store_true', default=True,
                    help='whether to use early stopping')
parser.add_argument('--patience', type=int, default=20,
                    help='patience')
parser.add_argument('--freeze_Ratio', type=float, default=0.5,
                    help='freeze ratio')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')

args = parser.parse_args()



if __name__ == '__main__':
    
    print("################# Settings ###################")
    
    Classic_Training(args)



  
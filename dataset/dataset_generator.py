#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:09:42 2022

@author: xuanyuanqiao

reference: https://github.com/KatherLab/HIA
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py
from PIL import Image
from torchvision import transforms
import glob

class Generic_WSI_Classification_Dataset(Dataset):

    def __init__(self, csv_path='',
                 data_dir='',
                 shuffle=False,
                 seed=1,
                 print_info=True,
                 label_dict={},
                 patient_strat=False,
                 label_col=None,
                 patient_voting="max"):

        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_inro = print_info
        self.patient_strat = patient_strat
        self.data_dir = data_dir

        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path)
        slide_data = self.Df_prep(data=slide_data, label_dict=self.label_dict, label_col=self.label_col)

        np.random.seed(seed)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        ##slide data prep

        self.slide_data = slide_data

        ##patient data prep

        patients = np.unique(np.array(self.slide_data["case_id"]))
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values

            if patient_voting == 'max':
                label = label.max()
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]

            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

        self.Cls_ids_prep()

    ##############################################################################

    def Df_prep(self, data, label_dict, label_col):

        if label_col != 'label':
            data['label'] = data[label_col].copy()

        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    ##############################################################################

    def Cls_ids_prep(self):

        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    ##############################################################################

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data["case_id"])

        else:
            return len(self.side_data)

    ##############################################################################

    def Return_splits(self, csv_path=None):

        assert csv_path
        all_splits = pd.read_csv(csv_path)

        train_split = self.Get_split_from_df(all_splits, 'train')
        val_split = self.Get_split_from_df(all_splits, 'val')
        test_split = self.Get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    ##############################################################################

    def Get_split_from_df(self, all_splits, split_key='train'):

        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['case_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)

        else:
            split = None

        return split

    def __Getitem__(self, idx):
        return None

    def Get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    ##############################################################################

    def Getlabel(self, ids):
        return self.slide_data['label'][ids]


##############################################################################
class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):

    def __init__(self, data_dir, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = True

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]

        full_path = os.path.join(self.data_dir, '{}.h5'.format(slide_id))

        with h5py.File(full_path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        features = torch.from_numpy(features)

        return features, label, coords


##############################################################################

class Generic_Split(Generic_MIL_Dataset):

    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)


##############################################################################
class Whole_Slide_Bag_new(Dataset):
    def __init__(self, file_path, pretrained=False, target_patch_size=-1):

        self.file_path = file_path
        self.raw_samples = glob.glob(file_path + '/*')

        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        self.roi_transforms = eval_transforms(pretrained=pretrained)


    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, index):

        image_path = self.raw_samples[index]
        temp = image_path.split('_(')
        temp = temp[-1].replace(').jpg', '')
        coord = [int(temp.split(',')[0]), int(temp.split(',')[1])]

        image = Image.open(image_path)
        if self.target_patch_size is not None:
            image = image.resize(self.target_patch_size)
            image = np.array(image)
        image = self.roi_transforms(image).unsqueeze(0)
        return image, coord


##############################################################################


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val


##############################################################################
def Summarize(args, labels, reportFile):
    print("label dictionary: {}\n".format(args.label_dict))
    reportFile.write("label dictionary: {}".format(args.label_dict) + '\n')
    print("number of classes: {}\n".format(args.n_classes))

    reportFile.write("number of classes: {}".format(args.n_classes) + '\n')

    for i in range(args.n_classes):
        print('Patient-LVL; Number of samples registered in class %d: %d\n' % (i, labels.count(i)))
        reportFile.write('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)) + '\n')
    print('-' * 30 + '\n')
    reportFile.write('-' * 30 + '\n')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 19:42:47 2022

@author: xuanyuanqiao

reference: https://github.com/KatherLab/HIA, https://github.com/mahmoodlab/CLAM

"""


import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import pandas as pd
import random
import argparse


from utils.core_utils import Train_AMIL
from eval.eval import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
from dataset.dataset_generator import Generic_MIL_Dataset, Summarize

##############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

##############################################################################

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

def get_value_from_key(d, key):
    
    values = [v for k, v in d.items() if k == key]
    if values:
        return values[0]
    return None  


def AMIL_training(args):
    
    random.seed(args.seed)
    
    folder = args.output_dir
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        

    split_dir = os.path.join(args.output_dir,"SPLITS")
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    args.result_dir = os.path.join(args.output_dir,"RESULT")
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
    
    reportFile  = open(os.path.join(args.output_dir,'Report.txt'), 'a', encoding="utf-8")
    reportFile.write('-' * 30 + '\n')
    reportFile.write(str(args))
    reportFile.write('-' * 30 + '\n')
    
    table = pd.read_csv(args.csvFile)
     
    patients= list(table["case_id"])
    patientsList = list(set(patients))

    labelsList = []

    for pt in patientsList:
        temp_tb = table.loc[table["case_id"] == pt]
        temp_label = list(temp_tb['label'])[0]
        labelsList.append(temp_label)
    
    yTrueLabel = labelsList
    le = preprocessing.LabelEncoder()
    yTrue = le.fit_transform(yTrueLabel)
    args.n_classes = len(set(yTrue))
    args.label_dict = dict(zip(le.classes_, range(len(le.classes_)))) 
    
    Summarize(args, list(yTrue), reportFile)
    
    
    ###########################################
    
    print('\nLoad the DataSet...')

        
    dataset = Generic_MIL_Dataset(csv_path = args.csvFile,
                        data_dir= args.feat_dir,
                        shuffle = False, 
                        seed = args.seed, 
                        print_info = True,
                        label_dict = args.label_dict,
                        patient_strat=True)
    
        

    ###########################################
    #######    Start Cross Validation   #######
    ###########################################
    
    print('IT IS A ' + str(args.k) + ' FOLD CROSS VALIDATION TRAINING !')
    patientID = np.array(patientsList)
    yTrue = np.array(yTrue)
    yTrueLabel = np.array(yTrueLabel)
    
    folds = args.k
    kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
    kf.get_n_splits(patientID, yTrue)
    
    foldcounter = 1
    
    for train_index, test_index in kf.split(patientID, yTrue):
        
        testPatients = patientID[test_index]   
        trainPatients = patientID[train_index] 
        
        testyTrue = yTrue[test_index]   
        trainyTrue = yTrue[train_index] 
        
        testyTrueLabel = yTrueLabel[test_index]   
        trainyTrueLabel = yTrueLabel[train_index]
        
        
        print('FOR TRAIN SET...\n') 
        
        train_data = pd.DataFrame(list(zip(trainPatients, trainyTrue, trainyTrueLabel)), columns = ['PATIENT', 'yTrue', 'yTrueLabel'])    
        print('FOR VALIDATION SET...\n')  
        val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.15))
        train_data = train_data[~train_data['PATIENT'].isin(list(val_data['PATIENT']))]           
        print('FOR TEST SET...\n')
        test_data = pd.DataFrame(list(zip(testPatients, testyTrue, testyTrueLabel)), columns = ['PATIENT', 'yTrue', 'yTrueLabel'])             
        
        train_data.reset_index(inplace = True, drop = True)
        test_data.reset_index(inplace = True, drop = True)
        val_data.reset_index(inplace = True, drop = True) 
        
        print('-' * 30)
        print("K FOLD VALIDATION STEP => {}".format(foldcounter)) 
        print('-' * 30)
        
        df = pd.DataFrame({'train': pd.Series(train_data['PATIENT']), 'test': pd.Series(test_data['PATIENT']), 'val' : pd.Series(val_data['PATIENT'])})
        df.to_csv(os.path.join(split_dir, 'TrainTestValSplit_{}.csv'.format(foldcounter)), index = False)                                                                       
        train_dataset, val_dataset, test_dataset = dataset.Return_splits(csv_path = os.path.join(split_dir, 'TrainTestValSplit_{}.csv'.format(foldcounter)))
        datasets = (train_dataset, val_dataset, test_dataset)
    
    
        model, results, test_auc  = Train_AMIL(datasets = datasets, fold = foldcounter, args = args, trainFull = False) 
        
        reportFile.write('AUC calculated by AMIL' + '\n')
        reportFile.write(str(test_auc) + '\n')
        reportFile.write('-' * 30 + '\n')
        
        patients = []
        filaNames = []
        yTrue_test = []
        yTrueLabe_test = []
        probs = {}
        
        for i_temp in range(args.n_classes):
            key = get_key_from_value(args.label_dict, i_temp)
            probs[key] = []
            
        for item in list(results.keys()):
            temp = results[item]
            patients.append(temp['case_id'])
            filaNames.append(temp['slide_id'])
            yTrue_test.append(temp['label'])
            yTrueLabe_test.append(get_key_from_value(args.label_dict, temp['label']))   
            
            for key in list(args.label_dict.keys()):
                probs[key].append(temp['prob'][0][get_value_from_key(args.label_dict, key)])
        
        probs = pd.DataFrame.from_dict(probs)                        
        df = pd.DataFrame(list(zip(patients, filaNames, yTrue_test, yTrueLabe_test)), columns =['PATIENT', 'FILENAME', 'yTrue', 'yTrueLabel'])
        df = pd.concat([df, probs], axis = 1)
        testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_SLIDE_BASED_FOLD_' + str(foldcounter) + '.csv')
        df.to_csv(testResultsPath, index = False)
        CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = foldcounter , attMil = True, reportFile = reportFile)
        reportFile.write('-' * 30 + '\n')                
        foldcounter +=  1
        
    
    patientScoreFiles = []
    slideScoreFiles = [] 
    
    for i in range(args.k):
        patientScoreFiles.append('TEST_RESULT_PATIENT_BASED_FOLD_' + str(i + 1) + '.csv')
        slideScoreFiles.append('TEST_RESULT_SLIDE_BASED_FOLD_' + str(i + 1) + '.csv')
        
    CalculateTotalROC(resultsPath = args.result_dir, results = patientScoreFiles, target_labelDict =  args.label_dict, reportFile = reportFile)
    reportFile.write('-' * 30 + '\n')
    MergeResultCSV(args.result_dir, slideScoreFiles, attMil = True)
    reportFile.close()
                    
    
##############################################################################
    
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI AMIL Training')
parser.add_argument('--feat_dir', type=str, default=None, 
                    help='feature directory')
parser.add_argument('--csvFile', type=str, default=None, 
                    help='annotation file')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=3, help='number of folds (default: 10)')
parser.add_argument('--output_dir', default='./results')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=0.25, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'fc','wce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
                
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'wce',None], default='svm',
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')

args = parser.parse_args()


settings = {'num_splits': args.k,
            'max_epochs': args.max_epochs, 
            'results_dir': args.output_dir,
            'lr': args.lr,
            'reg': args.reg,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'bag_weight': args.bag_weight,
            'inst_loss': args.inst_loss,
            'B': args.B}


if __name__ == '__main__':
    
    print("################# Settings ###################")
    
    for key, val in settings.items():
        print("{}:  {}".format(key, val)) 
    
    AMIL_training(args)



    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 01:39:58 2022

@author: xuanyuanqiao
"""

import os 
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torchvision import models
import json
import warnings
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
        
##############################################################################
       
def Print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


##############################################################################
            
def Collate_features(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return  [img, coords]

##############################################################################
            
def calculate_error(Y_hat, Y):
    
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

##############################################################################

def save_pkl(filename, save_object):
    
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

##############################################################################

def load_pkl(filename):
    
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file
        

###############################################################################

def Set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

###############################################################################
            
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

###############################################################################    

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


###############################################################################

def MakeBool(value):
    
    if value == 'True':
       return True
    else:
        return False
    
###############################################################################

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

###############################################################################

def isint(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

###############################################################################

def CheckForTargetType(labelsList):
    
    if len(set(labelsList)) >= 5:     
        labelList_temp = [str(i) for i in labelsList]
        checkList1 = [s for s in labelList_temp if isfloat(s)]
        checkList2 = [s for s in labelList_temp if isint(s)]
        if not len(checkList1) == 0 or not len (checkList2):
            med = np.median(labelsList)
            labelsList = [1 if i>med else 0 for i in labelsList]
        else:
            raise NameError('IT IS NOT POSSIBLE TO BINARIZE THE NOT NUMERIC TARGET LIST!')
    return labelsList
                    
###############################################################################            
    
def get_key_from_value(d, val):
    
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None   
    
 ###############################################################################   
    
def get_value_from_key(d, key):
    
    values = [v for k, v in d.items() if k == key]
    if values:
        return values[0]
    return None    
  
##############################################################################

def get_optim(model, args, params = False):
  
    if params:
      temp = model
    else:
      temp = filter(lambda p: p.requires_grad, model.parameters())
      
    if args.opt == "adam":
      optimizer = optim.Adam(temp, lr = args.lr, weight_decay = args.reg)
    elif args.opt == 'sgd':
      optimizer = optim.SGD(temp, lr = args.lr, momentum = 0.9, weight_decay = args.reg)
    else:
      raise NotImplementedError
      
    return optimizer
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 20:39:54 2022

@author: xuanyuanqiao

Reference :https://github.com/KatherLab/preProcessing
"""

from multiprocessing.dummy import Pool as ThreadPool
import stainNorm_Macenko
import multiprocessing
import os
import cv2
import numpy as np

global inputPath
global outputPath
global normalizer

##############################################################################

def Normalize_Main(item):
           
    outputPathRoot = os.path.join(outputPath, item)
    inputPathRoot = os.path.join(inputPath, item)
    inputPathRootContent = os.listdir(inputPathRoot)
    print()
    if not len(inputPathRootContent) == 0:
        print("Number of tiles : {} for file {}".format(len(inputPathRootContent),item))
        if not os.path.exists(outputPathRoot):
            os.mkdir(outputPathRoot)

            temp = os.path.join(inputPath, item)
            tempContent = os.listdir(temp)
            tempContent = [i for i in tempContent if i.endswith('.jpg')]
            for tempItem in tempContent:
                img = cv2.imread(os.path.join(inputPathRoot, tempItem))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                edge  = cv2.Canny(img, 40, 40)
                edge = edge / np.max(edge)
                edge = (np.sum(np.sum(edge)) / (img.shape[0] *img.shape[1])) * 100
                #print(edge)
                if edge > 2:
                    try:
                        nor_img = normalizer.transform(img)
                        cv2.imwrite(os.path.join(outputPathRoot, tempItem), cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
                        #print("Normalized image {}".format(os.path.join(outputPathRoot, tempItem)))
                    except:
                        print('Failed to normalize the tile {}.'.format(tempItem))


        print("Successfully processed WSI {}".format(item))
                    
##############################################################################
    
def Normalization(inputPath, outputPath, sampleImagePath, num_threads = 8):
    
    inputPathContent = os.listdir(inputPath)
    normPathContent = os.listdir(outputPath)
    
    remainlList = []
    for i in inputPathContent:
        if not i in normPathContent:
            remainlList.append(i)
            
    inputPathContent = [i for i in remainlList if not i.endswith('.bat')]
    inputPathContent = [i for i in inputPathContent if not i.endswith('.txt')]
    
    target = cv2.imread(sampleImagePath)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    global normalizer
    normalizer = stainNorm_Macenko.Normalizer()
    normalizer.fit(target)  

    pool = ThreadPool(num_threads)
    pool.map(Normalize_Main, inputPathContent)
    pool.close()
    pool.join()


###############################################################################


inputPath = "TCGA_COAD/WSI_result/BLOCKS"
outputPath = "TCGA_COAD/WSI_Normalized/"
sampleImagePath = "preprocessing/Ref.png"

Normalization(inputPath, outputPath, sampleImagePath, num_threads=8)




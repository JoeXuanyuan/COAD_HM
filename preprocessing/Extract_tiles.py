#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:50:04 2022

@author: xuanyuanqiao

reference: https://github.com/KatherLab/preProcessing

"""


## Import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.dummy import Pool as ThreadPool
from os.path import join, isfile, exists
import os
import progressbar
import numpy as np
import imageio
import argparse
import openslide as ops
import pandas as pd
import cv2



NUM_THREADS = 8

## run: python Extract_tiles.py -s INPUT -o OUTPUT --px PIXELS --ov OVERLAP --num_threads

###############################################################################
        
class SlideReader:

    
    def __init__(self, path, filetype, export_folder = None, pb = None):
        
        self.coord = []
        self.export_folder = export_folder
        self.pb = pb
        self.p_id = None
        self.extract_px = None
        self.shape = None
        self.basename = path.replace('.'+ path.split('.')[-1],'')
        self.name = self.basename.split('/')[-1]
        self.ignoredFiles = []
        self.noMPPFlag = 0
        
        if filetype in ["svs", "mrxs", 'ndpi', 'scn', 'tif']:
            self.slide = ops.OpenSlide(path)
        else:
            outputFile.write('Unsupported file type ' + filetype + ',' + path + '\n')
            return None
        
        # Load the image
        self.shape = self.slide.dimensions
        self.filter_dimensions = self.slide.level_dimensions[-1]
        self.filter_magnification = self.filter_dimensions[0] / self.shape[0]

        try:
            if ops.PROPERTY_NAME_MPP_X in self.slide.properties:
                self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
            elif 'tiff.XResolution' in self.slide.properties:
                self.MPP = 1 / float(self.slide.properties['tiff.XResolution']) * 10000
            else:
                self.noMPPFlag = 1
                outputFile.write('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
                return None
        except:
            self.noMPPFlag = 1
            outputFile.write('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
            return None
            

    def build_generator(self, size_px, size_um, stride_div, case_name, tiles_path,  category, fileSize, export = False):
                    
        self.extract_px = int(size_um / self.MPP)
        stride = int(self.extract_px * stride_div)
        
        slide_x_size = self.shape[0] - self.extract_px
        slide_y_size = self.shape[1] - self.extract_px
        
        for y in range(0, (self.shape[1]+1) - self.extract_px, stride):
            for x in range(0, (self.shape[0]+1) - self.extract_px, stride):
                is_unique = ((y % self.extract_px == 0) and (x % self.extract_px == 0))
                self.coord.append([x, y, is_unique])

        tile_mask = np.asarray([0 for i in range(len(self.coord))])
        self.tile_mask = None
        
        def generator():
            for ci in range(len(self.coord)):
                c = self.coord[ci]
                filter_px = int(self.extract_px * self.filter_magnification)
                if filter_px == 0:
                    filter_px = 1

                
                # Read the low-mag level for filter checking
                filter_region = np.asarray(self.slide.read_region(c, self.slide.level_count-1, [filter_px, filter_px]))[:, :, :-1]
                median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
                if median_brightness > 660:
                    continue

                # Read the region and discard the alpha pixels
                try:
                    region = np.asarray(self.slide.read_region(c, 0, [self.extract_px, self.extract_px]))[:, :, 0:3]
                    region = cv2.resize(region, dsize=(size_px, size_px), interpolation=cv2.INTER_CUBIC)
                except:
                    continue
                
                edge  = cv2.Canny(region, 40, 100)
                edge = edge / np.max(edge)
                edge = (np.sum(np.sum(edge)) / (size_px * size_px)) * 100
                
                if (edge < 4) or np.isnan(edge):   
                    continue 
                   
                tile_mask[ci] = 1
                coord_label = ci
                unique_tile = c[2]
                
                if stride_div == 1:
                    exportFlag = export and unique_tile
                else:
                    exportFlag = export
                    
                if exportFlag:                 
                    imageio.imwrite(join(tiles_path, case_name +'_('+str(c[0])+','+str(c[1])+').jpg'), region)
                yield region, coord_label, unique_tile

            if self.pb:
                if sum(tile_mask) <4:
                    outputFile.write('Number of Extracted Tiles < 4 ' + ',' + join(tiles_path, case_name)+ '\n')

                print('Remained Slides: ' + str(fileSize))
                print('***************************************************************************')
                    
            self.tile_mask = tile_mask
        return generator, slide_x_size, slide_y_size, stride

###############################################################################
        
class Convoluter:
    def __init__(self, size_px, size_um, stride_div, save_folder = ''):
        
        self.SLIDES = {}
        self.SIZE_PX = size_px
        self.SIZE_UM = size_um
        self.SAVE_FOLDER = save_folder
        self.STRIDE_DIV = stride_div
        
    def load_slides(self, slides_array, directory = "None", category = "None"):
        self.fileSize = len(slides_array)
        self.iterator = 0
        print('TOTAL NUMBER OF SLIDES IN THIS FOLDER : ' + str(self.fileSize))
        
        for slide in slides_array:
            name = slide.split('.')[:-1]
            name ='.'.join(name)
            name = name.split('/')[-1]
            filetype = slide.split('.')[-1]
            path = slide

            self.SLIDES.update({name: {"name": name,
                                       "path": path,
                                       "type": filetype,
                                       "category": category}})
    
        return self.SLIDES

    def convolute_slides(self):
        
        '''Parent function to guide convolution across a whole-slide image and execute desired functions.
        '''
        ignoredFile_list = []
        if not os.path.exists(join(self.SAVE_FOLDER, "BLOCKS")):
            os.makedirs(join(self.SAVE_FOLDER, "BLOCKS"))
            
        pb = progressbar.ProgressBar()
        pool = ThreadPool(NUM_THREADS)
        pool.map(lambda slide: self.export_tiles(self.SLIDES[slide], pb, ignoredFile_list), self.SLIDES)
        return pb, ignoredFile_list

    def export_tiles(self, slide, pb, ignoredFile_list):
        case_name = slide['name']
        category = slide['category']
        path = slide['path']
        filetype = slide['type']
        self.iterator = self.iterator + 1
        whole_slide = SlideReader(path, filetype, self.SAVE_FOLDER, pb=pb)

            
        if whole_slide.noMPPFlag:
            return
        
        tiles_path = whole_slide.export_folder + '/' + "BLOCKS"
        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)
            
        tiles_path = tiles_path + '/' + case_name
            
           
        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)
                 
         
        counter = len(os.listdir(tiles_path))
        if counter > 6:
           print("Folder already filled")
           print('***************************************************************************')
           return  
       
        gen_slice, _, _, _ = whole_slide.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name, tiles_path,  category, 
                                                         fileSize = self.fileSize - self.iterator, export=True)
        for tile, coord, unique in gen_slice():
            pass
        
###############################################################################
            
def get_args():
    
    parser = argparse.ArgumentParser(
        description='The script to generate the tiles for Whole Slide Image (WSI).')
    parser.add_argument(
        '-s', '--slide', help='Path to folder of images (SVS or JPG) to analyze.')
    parser.add_argument('-o', '--out',
                        help='Path to directory in which exported images and data will be saved.')
    parser.add_argument('--px', type=int, default=512,
                        help='Size of image patches to analyze, in pixels.')
    parser.add_argument('--ov', type = float, default = 1.0,
                    help='The Size of overlappig. It can be values between 0 and 1.')
    parser.add_argument('--um', type=float, default=255.3856,
                        help='Size of image patches to analyze, in microns.')
    parser.add_argument('--num_threads', type=int,
                        help='Number of threads to use when tessellating.')

    return parser.parse_args()

###############################################################################
    
if __name__ == ('__main__'):
        
    args = get_args()
    if not args.out:
        args.out = args.slide
    if args.num_threads:
        NUM_THREADS = args.num_threads        
    
    c = Convoluter(args.px, args.um, args.ov, args.out)
    

    slide_list = []
    for root, dirs, files in os.walk(args.slide):
        for file in files:
            if ('.ndpi' in file or '.scn' in file or 'svs' in file or 'tif' in file) and not 'csv' in file:
                fileType = file.split('.')[-1]
                slide_list.append(os.path.join(root, file))

    if os.path.exists(join(args.out, "BLOCKS")):
        temp = os.listdir(os.path.join(args.out, 'BLOCKS'))
        for item in temp:
            for s in slide_list:
                if item + '.' + fileType in s:
                    slide_list.remove(s)
                    
    ##load all slide images in given input folder
    c.load_slides(slide_list)

    pb = c.convolute_slides()
    
    ##compute tile number distribution 
    filter = []
    data = []
    L_num = 0
    H_num = 0
    out_folder = os.path.join(args.out, 'BLOCKS')
    
    for c, filename in enumerate(os.listdir(out_folder)):
        img_folder = os.path.join(out_folder,filename)
        num = 0
        for c, image in enumerate(os.listdir(img_folder)):
            if image.endswith(".jpg"):
                num+=1

        if num < 100:
            filter.append([filename, num])
            L_num += 1
        else:
            data.append([filename, num])
            H_num += 1
                
    filter_df = pd.DataFrame(filter, columns = ["slide", "tile_number"])
    data_df = pd.DataFrame(data, columns = ["slide", "tile_number"])
    csv_path_1 = args.out + 'filter_slildes.csv'
    csv_path_2 = args.out + 'left_slildes.csv'
    filter_df.to_csv(csv_path_1, index = False)
    data_df.to_csv(csv_path_2, index = False)
    
    global outputFile
    outputFile  = open(os.path.join(args.out,'report.txt'), 'a', encoding="utf-8")
    outputFile.write('The Features Selected For this Experiment: ' + '\n')
    outputFile.write('InputPath: ' + args.slide + '\n')
    outputFile.write('OutPutPath: ' + args.out + '\n')
    outputFile.write('Size of image patches to analyze, in pixels: ' + str(args.px) + '\n')
    outputFile.write('Number of WSI have less than 100 tiles: ' + str(L_num) + '\n')
    outputFile.write('Number of WSI have more than 100 tiles: ' + str(H_num) + '\n')
    outputFile.write('#########################################################################' + '\n')


    outputFile.close()
    
    
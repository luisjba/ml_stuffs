#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:54:34 2018

@author: luisjba
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def file_list(directory, extension='jpg'):
    """
    Function that return a list of file paths for files filtered by extension
        args: 
            directory: the string direcotory to list in
        return:
            list: list of every file found with the extesion provided

    """
    return [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and f.endswith(".{}".format(extension))]

def extract_dataset_hand_sings(directory, extension='jpg'):
    """
    Function to extract the images and label from a directory, the class is extracted from the fist caracter of the image file name 
    for example the image 3_IMG_6110.jpg has the class number '3', extracted form the fist cacracter in the file name. The functions was
    programed for the hand sings dataset 
        args:
            directory: the string directory that contains the dataset image 
        return:
            images: list of images file directories
            labels: list of integer labels   
    Example: 
        images, labels = extract_dataset_hand_sings('dataset/signs')
    """
    images = file_list(directory, extension)
    lables = [int(img.split('/')[-1][0]) for img in images]
    return images, lables

def images_to_nparray(images, scale=True):
    """
    Function to read images from files and set into an array list
        args: 
            images: a list of images files apth to read from
            scale: If True, the image will be scaled the RGB 255 to 0-1
        return:
            images: nparray of images as float 32 values
    Example:
        images = images_to_nparray(images)
    """
    #Read the images
    images = [plt.imread(img) for img in images]
    # COvert to numpy array
    images = np.asarray(images).astype(np.float32, casting='safe')
    if scale:
        # Scale
        images /= 255.0
    return images
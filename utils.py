#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:54:34 2018

@author: luisjba
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

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

def labels_to_nparray(labels):
    """
    Function to convert a list of numbers to numpy array
        args:
            labels: the list of labels
        return:
            numpy array 
    """
    return np.hstack(labels)

def convert_numpy_to_torch_tensors_features_matrix(data, label_encoders = []):
    """
        Function to convert numpy dataset with the data grouped by main identifier in the first column and the  and features
        in the second column. The data value for each feature is in the third column.
        The resulted torch tensor is shape is n = number of rows and m = number of diferent features.
        
        args:
            data: 
                Numpy array or dataframe containing the data to convert to torch tensrors. Te column 0 is the group column main identifier, and represent the differentes
                train examples. The column 1 represent the dimensions or features. Finally the column 2 is the dependent value or the "y" variable for each feature belonging to each main identifier
            label_encoders: 
                Represent the encoders used to determine the shape of the matrix, the firt encoder is for the number of rows 
                and second encoder for the number of columns or features in the matrix    
        return: 
            Torch tensor with the shape n=size encoder 0 m= size of encoder 1 (features)
    """
    if len(label_encoders) < 2:
        raise ValueError('The parameter label_encoders must contains 2 items of LabelEncoder')
    if len([e for e in label_encoders if type(e) is not LabelEncoder]) > 0 :
        raise ValueError('All the items in the label_encoders list parameter must be of type LabelEncoder')
    m_rows = len(label_encoders[0].classes_) # Length or size for Dimension 1, the number of rows in the matrix
    m_cols = len(label_encoders[1].classes_)# Length or size for Dimension 2 ot the number of cols in the matrix
    m_tensors = torch.zeros(m_rows, m_cols, dtype=torch.float)
    for i_row in range(m_rows):
        data_found = data[:,[1,2]][data[:,0] == i_row] #columns filtered by index value for dimension 1
        if (data_found.shape[0] > 0) : # check that have data
            m_tensors[i_row,data_found[:,0]] = torch.FloatTensor(data_found[:,1]) # set the values to the corresponding columns
    return m_tensors
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

def is_numeric(obj):
    "Function to check if an object is numeric"
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

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

def convert_numpy_to_torch_tensor_features_matrix(data, feature_label_encoder):
    """
        Function to convert numpy dataset with the data grouped by optional filter in the 0 index column and feature 
        in the next column (or 0 if not filter provided in the index 0 column).
        The 'y' or dependent value is taken from the last index column form the data
        The resulted torch tensor is shape is (n,m) where: 
            n = 1 if not filter data provided or the length of diferents classes in the filter_label_encoder provided or generated 
                from the filter data
            m = number of diferent features or classes in the feature_label_encoder.
        
        args:
            data: 
                Numpy array or dataframe containing the data to convert to torch tensrors. Te column 0 is the group column main identifier, and represent the differentes
                train examples. The column 1 represent the dimensions or features. Finally the column 2 is the dependent value or the "y" variable for each feature belonging to each main identifier
            feature_label_encoder: 
                The Label encoder for the features classes to contruct the tensor matrix
        return: 
            Torch tensor matrix with shape (n,m)
    """
    if type(feature_label_encoder) is not LabelEncoder:
        raise ValueError('The parameter feature_label_encoder must be a LabelEncoder object type')
    if len(data.shape) == 1:
        # reshape to matrix ( size/2, size/2 ) if it is a vector
        data = data.reshape(( int(data.shape[0]/2), int(data.shape[0]/2) ))
    if data.shape[1] < 2:
        raise ValueError('Yo must provide at least 2 columns  containing the categorical values and dependent value in the last column')
    y_index = data.shape[1] - 1
    categorical_index = 0 if y_index == 1 else 1
    # Transform the features value
    if not is_numeric(data[:,categorical_index][0]):
        data[:,categorical_index] = feature_label_encoder.transform(data[:,categorical_index])
    rows = 1
    filter_label_encoder = LabelEncoder()
    if categorical_index == 1 :
        filter_label_encoder.fit(data[:,0])
        data[:,0] = filter_label_encoder.transform(data[:,0])
        rows = len(filter_label_encoder.classes_)
    filter_data_columns = [0,1,y_index] if  categorical_index == 1 else [0,y_index]
    # Select the important columns and convert to Float data type
    data = np.array(data[filter_data_columns], dtype = np.float)
    # Recalculate y_index
    y_index = data.shape[1] - 1
    n_features = len(feature_label_encoder.classes_)
    X = torch.zeros(rows, n_features, dtype=torch.float)
    selected_columns = [categorical_index, y_index]
    for i_row in range(rows):
        if categorical_index == 1: 
            # Filter by row index
            data_found = data[:,selected_columns][data[:,0] == i_row]
        else: 
            # no filter to apply
            data_found = data[:,selected_columns]  
        if (data_found.shape[0] > 0) : # check that have data
            features_found_columns_index = data_found[:,0].astype(np.int)
            y_found_values = data_found[:,1]
            if (len(features_found_columns_index) > 0) :
                if not y_found_values.dtype.name == 'float64':
                    print('convert {} to float'.format(y_found_values.dtype.name))
                    y_found_values = y_found_values.astype(np.float)
                y_found_values = torch.FloatTensor(y_found_values)
                X[i_row,features_found_columns_index] = y_found_values
    return X
        
    





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:54:34 2018
Last Updated on Thu Nov 29

@author: luisjba
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

torch_optimizer_list = [op for op in dir(optim) if  re.match('^(?!_|Optimizer)^[A-Z]+',op)] #Seatch for the avaliable options in the optim module
torch_activation_list = [a for a in dir(nn.modules.activation) if re.match('^(?!_|Module)^[A-Z]+\w+$',a)]
torch_loss_list = [l for l in dir(nn.modules.loss) if re.match('^(?!_)^[A-Z]+.*(Loss)$',l)]

def torch_optimizer_name(optimizer):
    """
    Funtion to get the name ot the torch optimizer based on the posible 
    list ['ASGD','Adadelta','Adagrad','Adam','Adamax','LBFGS','RMSprop','Rprop','SGD','SparseAdam']
        args:
            optimizer: torch.optim.Optimizer object
        return: str optimizer name
    """
    optimizer_name = None
    for op in torch_optimizer_list :
        if isinstance(optimizer, eval("optim.{}".format(op))):
            optimizer_name = op.lower()
            break
    if optimizer_name is not None:
        return optimizer_name
    else :
        raise ValueError('Could found torch optimizer name optimizer object:', optimizer) 
    
def torch_optimizer_get(name, parameters, **arg_dict):
    """
    Function to get the the optimizer object found by name and applying the arguments dict
        args:
            name: The name of the optimizer
            parameters: the Model Parameters
            **arg_dict: the arguments dict to pass to the optimizer
        return: torch.optim.Optimizer object
    """
    optimizer = None
    for op in torch_optimizer_list :
        if op.lower() == name.lower():
            optimizer = eval("optim.{}".format(op))
            break
    if optimizer is not None:
        return optimizer(parameters, **arg_dict)
    else :
        raise ValueError('Could not interpret torch optimizer function name:', name) 

def torch_activation_name(activation):
    """
    Funtion to get the name ot the torch module activation based on the posible 
    list names ['ELU','GLU','Hardshrink','Hardtanh','LeakyReLU','LogSigmoid','LogSoftmax',
    'PReLU','RReLU','ReLU','ReLU6','SELU','Sigmoid','Softmax','Softmax2d','Softmin','Softplus',
    'Softshrink','Softsign','Tanh','Tanhshrink','Threshold'] 
        args:
            activation: torch.modules.Module object
        return: str Module activation name
    """
    activation_name = None
    for a in torch_activation_list :
        if isinstance(activation, eval("nn.modules.activation.{}".format(a))):
            activation_name = a.lower()
            break
    if activation_name is not None:
        return activation_name
    else :
        raise ValueError('Could found activation name activation object:', activation)
    
def torch_activation_get(name):
    """
    Function to get the the activation Module object found by name
        args:
            name: The name of the activation Module
        return: torch.modules.Module object
    """
    activation = None
    for a in torch_activation_list :
        if a.lower() == name.lower():
            activation = eval("nn.modules.activation.{}".format(a))
            break
    if activation is not None:
        return activation()
    else :
        raise ValueError('Could not interpret torch Module activation function name:', name)   
        
def torch_loss_name(loss):
    """
    Funtion to get the name ot the torch Module loss based on the posible 
    list names ['BCELoss','BCEWithLogitsLoss','CosineEmbeddingLoss','CrossEntropyLoss',
    'HingeEmbeddingLoss','KLDivLoss','L1Loss','MSELoss','MarginRankingLoss',
    'MultiLabelMarginLoss','MultiLabelSoftMarginLoss','MultiMarginLoss','NLLLoss',
    'PoissonNLLLoss','SmoothL1Loss','SoftMarginLoss','TripletMarginLoss']
        args:
            activation: torch.modules.loss.{lossClassName}  object
        return: str Module Loss loss name
    """
    loss_name = None
    for l in torch_loss_list :
        if isinstance(loss, eval("nn.modules.loss.{}".format(l))):
            loss_name = l.replace("Loss","").lower()
            break
    if loss_name is not None:
        return loss_name
    else :
        raise ValueError('Could found torch Module Loss object:', loss) 
    
def torch_loss_get(name):
    """
    Function to get the the torch  Module loss object found by name
        args:
            name: The name of the loss Module
        return: torch.modules.loss.{lossClassName} object
    """
    loss = None
    loss_name = name.lower()
    if not re.match('(loss)$',loss_name):
        loss_name += "loss"
    for l in torch_loss_list :
        if l.lower() == loss_name.lower():
            loss = eval("nn.modules.loss.{}".format(l))
            break
    if loss is not None:
        return loss()
    else :
        raise ValueError('Could not interpret torch Module Loss function name:', name)

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
    y_data = data[:,-1]
    X_data = data[:,0:data.shape[1]-1]
    do_filter = X_data.shape[1] > 1
    categorical_index = 0 if not do_filter else 1
    # Transform the features value
    if not is_numeric(X_data[:,categorical_index][0]):
        X_data[:,categorical_index] = feature_label_encoder.transform(X_data[:,categorical_index])
    rows = 1
    filter_label_encoder = LabelEncoder()
    if do_filter :
        X_data[:,0] = filter_label_encoder.fit_transform(X_data[:,0])
        rows = len(filter_label_encoder.classes_)
    n_features = len(feature_label_encoder.classes_)
    X = torch.zeros(rows, n_features, dtype=torch.float)
    for i_row in range(rows):
        data_filter = X_data[:,0] == i_row if do_filter else np.zeros_like(X_data.shape[0])
        features = X_data[:,0] if not do_filter else X_data[:,1][data_filter]
        if (features.shape[0] > 0) : # check that have data
            features = features.astype(np.int)
            features_values = y_data[:] if not do_filter else y_data[data_filter]
            features_values = features_values.astype(np.float)
            X[i_row,features] = torch.FloatTensor(features_values)
    return X
        
    





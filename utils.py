#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:54:34 2018

@author: luisjba
"""
import os

def file_list(directory, extension='jpg'):
    """Funtion that return a list of file paths for files filtered by extension
        args: 
            directory: the string direcotory to list in

    """
    return [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and f.endswith(".{}".format(extension))]


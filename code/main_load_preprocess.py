# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:34:50 2018

@author: gilbe
"""
import numpy as np
import pandas as pd
from utils.utils import PROJECT_DATA_DIR
import os
import time
import mxnet as mx
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler,
                                   Imputer,
                                   QuantileTransformer)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras import optimizers
from sklearn.metrics import f1_score
from load_preprocess import (load_data,
                             get_xy,
                             scale_data,
                             binarize_y)
"""File to verify and run code for big data sets"""

global train
global test


def  main():
    #print('Loading data...')
    #train = load_data()
    #test = load_data(file='xab_test_400.csv')
    #print('Done with loading data.')
    start = time.time()

    xtrain, ytrain = get_xy(train)
    xtest, ytest = get_xy(test)
    print('Binarization of ytrain, ytest.')
    ytrain_bin = binarize_y(ytrain)
    ytest_bin = binarize_y(ytest)
    print('info xtrain:')
    print('xtrain.shape:', xtrain.shape, 'ytrain.shape:', ytrain_bin.shape)
    print('xtest.shape:', xtest.shape, 'ytest.shape:', ytest_bin.shape)

    print('unique ytrain values:', ytrain.unique())
    print('unique ytest values:', ytest.unique())

    print('unique ytrain values:', ytrain_bin.unique())
    print('unique ytest values:', ytest_bin.unique())

    print('')
    print('Look at both training and test sets.')
    print(xtrain.head())
    print(xtest.head())
    print('')
    print('Scale data: ')
    xtrain_sc, xtest_sc = scale_data(xtrain, xtest, scaler_mode=None)
    print('shape of xtrain_sc:', xtrain_sc.shape)
    print('shape of xtest_sc:', xtest_sc.shape)
    print('xtrain_sc:', xtrain_sc)
    print('')
    print('xtest_sc:', xtest_sc)
    end = time.time()
    print('')
    print('Elapsed time:', end - start)



if __name__ == '__main__':
    main()
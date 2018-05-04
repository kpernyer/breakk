# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:54:03 2018

@author: gilbe
"""

import numpy as np
import pandas as pd
from utils.utils import PROJECT_DATA_DIR
import os
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



def load_data(file='xab.csv'):
    """ Data is supposed to be preprocessed
    and space separated """
    data = pd.read_csv(os.path.join(PROJECT_DATA_DIR, file),
                       sep='\s+', header=None)
    return data


def get_xy(data):
    """ We suppose the y labels are in the
    last column in the data set """
    y = data[data.columns[-1]]
    x = data.drop(data.columns[-1], axis=1)

    return x, y


def scale_data(xtrain, xtest, scaler_mode):
    """ Args
    ----
    xtrain: training dataframe with features

    xtest: test dataframe with features

    scaler: a scikit-learn scaler, either minmax,
    or normalization"""

    if scaler_mode == 'minmax':
        scaler = MinMaxScaler()

    elif scaler_mode == 'standardscaler':
        scaler = StandardScaler()

    else:
        scaler = QuantileTransformer(output_distribution='normal')

    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    return xtrain, xtest

def binarize_y(y, arg_list=[12, 13, 17, 20]):
    """ pass list to take values in one of 2 categories """
    return y.map(lambda x: 1 if x in arg_list else 0)


if __name__ == '__main__':
    print('Loading data...')
    data = load_data()
    print('Done with loading data.')


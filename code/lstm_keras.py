# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:16:09 2018

@author: gilbe
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras import optimizers
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from load_preprocess import (load_data,
                             get_xy,
                             scale_data,
                             binarize_y,
                             prepare_data)

""" Long Short Term Memory model """

def train_lstm(xtrain, ytrain, xtest, ytest,
               epochs=20,
               batch_size=2**8,
               units=[1200, 500, 75],
               input_dim=1200,
               drop=0.4, lr=0.0001,
               binary_class=True):

    model = Sequential()
    model.add(LSTM(units=20, input_shape=(400, 3)))
    model.add(BatchNormalization(input_shape=(400, 3)))
    model.add(Dropout(0.4))
    model.add(LSTM(units=20))
    model.add(BatchNormalization(input_shape=(400, 3)))
    model.add(Dropout(0.4))
    model.add(Dense(units=11, activation='softmax'))

    optim = optimizers.Adam(lr=lr,
                            beta_1=0.9,
                            beta_2=0.999,
                            decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    history = model.fit(xtrain,
                        ytrain,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(xtest, ytest),
                        shuffle=False)
    return model, history


if __name__ == '__main__':
    train = load_data(file='all_training_400_minisensor_1.csv')
    test = load_data(file='all_test_400_minisensor.csv')

    xtrain, ytrain, xtest, ytest = prepare_data(train, test)
    xtrain_sc, xtest_sc = scale_data(xtrain, xtest)





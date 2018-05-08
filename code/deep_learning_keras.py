# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:11:31 2018

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
                             binarize_y)


"""Deep learning model"""


def plot_loss(history):
    plt.figure(figsize=(7, 7))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def train_model(xtrain, ytrain, xtest, ytest,
                epochs=20,
                batch_size=2**9,
                units=[1200, 500, 75],
                input_dim=1200,
                drop=0.4, lr=0.0001):

    optim = optimizers.Adam(lr=lr,
                        beta_1=0.9,
                        beta_2=0.999,
                        decay=1e-6)

    model = Sequential()
    model.add(Dense(units=units[0],
                    activation='relu',
                    input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Dense(units[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Dense(units=units[2], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Dense(units=1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    history = model.fit(xtrain, ytrain,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(xtest, ytest))
    return model, history


if __name__ == '__main__':
    train = load_data(file='all_training_400_minisensor_1.csv')
    test = load_data(file='all_test_400_minisensor.csv')

    xtrain, ytrain = get_xy(train)
    xtest, ytest = get_xy(test)
    ytrain_bin = binarize_y(ytrain)
    ytest_bin = binarize_y(ytest)
    xtrain_sc, xtest_sc = scale_data(xtrain, xtest, scaler_mode=None)
    print('train.shape:', train.shape)
    print('test.shape:', test.shape)
    print('xtrain_sc.shape:', xtrain_sc.shape)
    print('xtest_sc.shape:', xtest_sc.shape)

    model, history = train_model(xtrain_sc, ytrain_bin,
                                 xtest_sc, ytest_bin)

    pred_train = model.predict_classes(xtrain_sc)
    pred_test = model.predict_classes(xtest_sc)
    print('Training score:', f1_score(ytrain_bin, pred_train))
    print('Test score:', f1_score(ytest_bin, pred_test))

    plot_loss(history)


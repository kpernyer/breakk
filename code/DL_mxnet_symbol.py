# -*- coding: utf-8 -*-
"""
Created on Mon May 21 08:55:02 2018

@author: gilbe
"""

"""MXNet symbol API
transformers can be implemened in mxnet to use
gpu computations on the tensor and speed calculations
at the same time mxnet supports composition of transformers

"""
import logging
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import pandas as pd
from utils.utils import PROJECT_DATA_DIR
import os
import mxnet as mx
import mxnet.ndarray as nd
from time import time
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from load_preprocess import (load_data,
                             get_xy,
                             scale_data,
                             binarize_y,
                             prepare_data)
import warnings
warnings.filterwarnings("ignore",  category=DeprecationWarning)
#ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()



def train_dnn(train_iter, val_iter,
              hidden_units=[1200, 500, 75],
              num_outputs=1, batch_size=2**9):
    """ ---Model for binary classification---
    TODO: implement the layers with a for loop, see
    the example jupyter notebook. Also remember to use
    enumerate in for loop to get the indeces and values
    in hidden_units
    """
    # set the context on GPU is available otherwise CPU

    train_iter.reset()
    val_iter.reset()

    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(
            net,
            name='fc1',
            num_hidden=hidden_units[0])

    net = mx.sym.Activation(
            net,
            name='relu1',
            act_type='relu')

    net = mx.sym.FullyConnected(
            net,
            name='fc2',
            num_hidden=hidden_units[1])

    net = mx.sym.Activation(
            net,
            name='relu2',
            act_type='relu')

    net = mx.sym.FullyConnected(
            net,
            name='fc3',
            num_hidden=hidden_units[2])

    net = mx.sym.Activation(
            net,
            name='relu3',
            act_type='relu')

    net = mx.sym.FullyConnected(
            net,
            name='out',
            num_hidden=num_outputs)

    net = mx.sym.LogisticRegressionOutput(
            net,
            name='softmax')

    mod = mx.mod.Module(net, context=mx.gpu())
    # pass shapes of iterators to allocate space
    mod.bind(data_shapes=train_iter.provide_data,
            label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier())
    mod.init_optimizer(
            optimizer='Adam',
            optimizer_params=(('learning_rate', 0.01), ))

    mod.fit(train_data=train_iter,
            eval_data=val_iter,
            optimizer='Adam',
            optimizer_params={'learning_rate': 0.01},
            eval_metric='acc',
            num_epoch=20,
            batch_end_callback = mx.callback.Speedometer(batch_size, 100),
            )


# Test implementing lstm in mxnet
def train_lstm(train_iter, val_iter,
               hidden_units=[1200, 500, 75],
               num_outputs=1):
    net_lstm = mx.sym.Variable('data')
    weight = mx.sym.Variable('weight', init=mx.init.Zero())
    bias = mx.sym.Variable('bias', init=mx.init.Zero())
#    rnn_h_init = mx.sym.Variable('LSTM_init_h')
#    rnn_c_init = mx.sym.Variable('LSTM_init_c')
#    rnn_params = mx.sym.Variable('LSTM_bias')

    num_hidden = 20
    num_lstm_layer = 1

#    net = mx.sym.transpose(net, (1, 0, 2))
    # maybe use mx.sym.transpose(data, axes=(1, 0, 2)) ?? (time, batch, columns)
    net_lstm = mx.sym.RNN(
            net_lstm,
            num_layers=num_lstm_layer,
            state_size=num_hidden,
            name='rnn_lstm1',
            mode='lstm',
#            state=rnn_h_init,
#            state_cell=rnn_c_init,
            parameters=weight,
            p=0.4)

    net_lstm = mx.sym.FullyConnected(
            net_lstm,
            name='out',
            num_hidden=num_outputs)

    net_lstm = mx.sym.LogisticRegressionOutput(
            net_lstm,
            name='softmax')

    mod = mx.mod.Module(net_lstm, context=mx.gpu())

    mod.bind(data_shapes=train_iter.provide_data,
             label_shapes=train_iter.provide_label)

    """Xavier not accepted for initialization in LSTM"""
    mod.init_params(mx.initializer.Uniform(scale=1.0))

    mod.init_optimizer(optimizer='Adam',
                       optimizer_params=(('learning_rate', 0.01)))

    metric = mx.metric.create('acc')
    for epoch in range(20):
        train_iter.reset()
        for batch in train_iter:
            print('shape of batch:', batch.data[0].shape)
            print(batch.data[0].asnumpy())
            break
            predictions = mod.forward(batch, is_train=True)
#            mod.metric_updeta(labels=)
            mod.backward()
            mod.update()
        print('Epoch %d, Training %s' % (epoch))



#    mod.fit(train_data=train_iter,
#            eval_iter=val_iter,
#            optimizer='Adam',
#            optimizer_params={'learning_rate': 0.01},
#            eval_metric='acc',
#            num_epoch=20)


def main():
    train = load_data(file='all_training_400_minisensor_1.csv')
    test = load_data(file='all_test_400_minisensor.csv')
    xtrain, ytrain, xtest, ytest = prepare_data(train, test, binary_class=True)
    xtrain_sc, xtest_sc = scale_data(xtrain, xtest)
    print(xtrain.head())
    print('')
    print(ytrain.head())
    print('xtrain.shape:', xtrain.shape)
    print('xtest.shape:', xtest.shape)
    print('ytrain.shape:', ytrain.shape)
    print('ytest.shape:', ytest.shape)

    xtrain_mx = mx.nd.array(xtrain_sc, dtype=np.float32)
    ytrain_mx = mx.nd.array(ytrain.reshape(-1, 1))
    xtest_mx = mx.nd.array(xtest_sc, dtype=np.float32)
    ytest_mx = mx.nd.array(ytest.reshape(-1, 1))
    batch_size=2**9

    train_iter = mx.io.NDArrayIter(
            xtrain_mx,
            ytrain_mx,
            batch_size=batch_size,
            shuffle=True)

    val_iter = mx.io.NDArrayIter(
            xtest_mx,
            ytest_mx,
            batch_size=batch_size)

    train_dnn(train_iter, val_iter)

    xtrain_lstm = xtrain.values.reshape(-1, 3)
    xtest_lstm = xtest.values.reshape(-1, 3)
    scaler = QuantileTransformer(output_distribution='normal')
    xtrain_lstm_sc = scaler.fit_transform(xtrain_lstm)
    xtest_lstm_sc = scaler.transform(xtest_lstm)

    print(xtrain_lstm_sc.shape)
    print(xtest_lstm_sc.shape)
    """ Change time steps from 400 to 20 to test if this is the problem"""
    xtrain_lstm_sc = mx.nd.array(xtrain_lstm_sc.reshape(-1, 400, 3))
    val_lstm_sc = mx.nd.array(xtest_lstm_sc.reshape(-1, 400, 3))
    print('shape of xtrain_lstm_sc:', xtrain_lstm_sc.shape)

    # transform to mxnet array
    train_lstm_iter = mx.io.NDArrayIter(
            xtrain_lstm_sc,
            ytrain_mx,
            batch_size,
            shuffle=True,
            last_batch_handle='discard')

    val_lstm_iter = mx.io.NDArrayIter(
            val_lstm_sc,
            ytest_mx,
            batch_size,
            shuffle=False,
            last_batch_handle='discard')

    print('Start training lstm with mxnet...')
    train_lstm(train_lstm_iter, val_lstm_iter)


if __name__ == '__main__':
    main()





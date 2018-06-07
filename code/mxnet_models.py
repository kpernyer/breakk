# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:12:04 2018

@author: gilbe
"""
import numpy as np
import os
import mxnet as mx
import warnings
warnings.filterwarnings("ignore",  category=DeprecationWarning)
from sklearn.metrics import f1_score, accuracy_score



"""Hybrid model for better performance than gluon"""
from mxnet.gluon import nn
from mxnet import gluon


def HybridNet(gloun):
    """ Start testing a small model,
    if we pass a symbol variable then the model
    is static and fast """
    def __init__(self, drop=0.4, num_outputs=1,**kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            self.h = [1200, 500, 75] # hidden units
            self.drop = drop
            self.num_outputs = num_outputs
            self.fc1 = nn.Dense(self.h[0])
            self.fc2 = nn.Dropout(self.drop)
            self.fc3 = nn.Dense(self.h[1])
            self.fc4 = nn.Drop(self.drop)
            self.fc5 = nn.Dense(self.drop)
            self.fc6 = nn.Dropout(self.drop)
            self.fc7 = nn.Dense(num_outputs)

    def hybrid_forward(self, F, x):
        """ F: backend """
        x = F.relu(self.fc1(x))
        x = F.drop(x, self.drop)
        return self.fc7(x)


""" Multilayer perceptron"""
def nn_model(num_outputs=1,
             hidden_layers=[1200, 500, 75]):
    data = mx.sym.Variable('data')
    net = data
    for i, units in enumerate(hidden_layers):
        with mx.name.Prefix('layer%d_' % (i + 1)):
            net = mx.sym.FullyConnected(
                    data=net,
                    name='fc',
                    num_hidden=units)
            net = mx.sym.Activation(
                    net,
                    name='relu',
                    act_type='relu')
    net = mx.sym.FullyConnected(
            net,
            name='output',
            num_hidden=num_outputs)
    net = mx.sym.LogisticRegressionOutput(
            net,
            name='softmax')
    return net


""" RNN models """
def rnn_fused(timesteps=400,
              num_layers=3,
              mode='lstm',
              num_hidden=20,
              dropout=0.4,
              num_outputs=1,
              batch_size=2**9,
              input_dim=3):

    data = mx.sym.Variable('data')
    """ Reshape input """
    input_shape = (timesteps, batch_size, input_dim)
    data = mx.sym.Reshape(data, shape=input_shape)

    """num_hidden: number of units in output symbol"""
    for i in range(num_layers):
        """ Check if data is flowing correctly trough
        the network"""
        outputs = data # this ensures right data flows
                        # through the network
        fused_lstm_cell = mx.rnn.FusedRNNCell(
            num_hidden=num_hidden,
            dropout=dropout)
        """ Implement many layers with for-loop as it is
        more effective when using multiple gpus"""
        outputs, _  = fused_lstm_cell.unroll(
            length=timesteps,
            inputs=outputs,
            merge_outputs=True)
        """ Reshape output from LSTM"""
    output_shape = (batch_size, timesteps, num_hidden)
    outputs = mx.sym.Reshape(outputs, shape=output_shape)
    outputs = mx.sym.Dropout(outputs, p=dropout)
    outputs = mx.sym.FullyConnected(
            data=outputs,
            name='out',
            num_hidden=num_outputs)
    outputs = mx.sym.LogisticRegressionOutput(
            outputs,
            name='softmax')

    return outputs


def train_lstm(xtrain_mx,
               ytrain_mx,
               xval_mx,
               yval_mx,
               timesteps=400,
               num_layers=3,
               mode='lstm',
               num_hidden=20,
               dropout=0.4,
               num_outputs=1,
               batch_size=2**9,
               input_dim=3,
               learning_rate=0.01,
               num_epoch=20):

    """ ---Args---
    xtrain_mx: training data a mxnet NDArray tensor
    ytrain_mx: training labels, a mxnet NDArray tensor
    xval_mx: validation set, a mxnet NDArray tensor
    yval_mx: test labels, a mxnet NDArray tensor
    timesteps: length for sequence
    num_layers: number of hidden layers
    mode: type of network, eg LSTM, GRU, RNN
    num_hidden: number of hidden units
    dropout: dropout rate for regularization
    num_outputs: no of outputs
    batch_size: batch size
    input_dim: no of features
    learning_rate: learning rate
    num_epoch: no of epochs to train
    """

    train_iter = mx.io.NDArrayIter(
            xtrain_mx,
            ytrain_mx,
            batch_size,
            shuffle=True,
            last_batch_handle='discard')

    val_iter = mx.io.NDArrayIter(
            xval_mx,
            yval_mx,
            batch_size,
            shuffle=False,
            last_batch_handle='discard')

    train_iter.reset()
    val_iter.reset()

    net = rnn_fused(timesteps=timesteps,
                    num_layers=num_layers,
                    num_hidden=num_hidden,
                    dropout=dropout,
                    num_outputs=num_outputs,
                    batch_size=batch_size,
                    input_dim=input_dim)

    mod = mx.mod.Module(net, context=mx.gpu())
    mod.bind(data_shapes=train_iter.provide_data,
             label_shapes=train_iter.provide_label)

    mod.init_params(initializer=mx.init.Xavier())
    mod.init_optimizer(
            optimizer='sgd',
            optimizer_params=(('learning_rate', learning_rate), ))

    mod.fit(train_data=train_iter,
            eval_data=train_iter,
            eval_metric='f1',
            num_epoch=num_epoch,
            batch_end_callback = mx.callback.Speedometer(
                    batch_size, 100))

# =============================================================================
#     # score returns a list with one elemnet
#     # which contains a tuple with 2 elements index[0][1]
#     f1_train = mod.score(train_iter, 'f1')
#     f1_test = mod.score(val_iter, 'f1')
# =============================================================================

    return mod, train_iter, val_iter
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:59:22 2019

@author: wmy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation, Dense
from keras.models import Model
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization
from keras.layers import Conv1D, ZeroPadding1D, AveragePooling1D, Flatten
from keras.layers import LeakyReLU, concatenate, Reshape, Softmax
from keras.layers import Dropout
from IPython.display import SVG
from keras.utils import plot_model
from keras import layers
from PIL import Image

def DenseBlock(X, nF, nG, nD):
    for i in range(nD):
        T = BatchNormalization(axis=2)(X)
        T = LeakyReLU(alpha=0.1)(T)
        T = Conv1D(filters=nF, kernel_size=1, strides=1, padding='valid')(T)
        T = BatchNormalization(axis=2)(T)
        T = LeakyReLU(alpha=0.1)(T)
        T = ZeroPadding1D(padding=1)(T)
        T = Conv1D(filters=nG, kernel_size=3, strides=1, padding='valid')(T)
        X = concatenate([X, T], axis=2)
        nF += nG
        pass
    return X

def ResidualDenseBlock(X, nC_in, nC_out, nF, nG, nD, strides=1):
    branch = DenseBlock(X, nF, nG, nD)
    branch = Conv1D(filters=nC_out, kernel_size=1, strides=strides, padding='valid')(branch)
    if nC_in != nC_out or strides != 1:
        X = Conv1D(filters=nC_out, kernel_size=1, strides=strides, padding='valid')(X)
        pass
    X = Add()([branch, X])
    return X

def DeepPianoModel(step=128, ndim=88):
    X_in = Input((step, ndim))
    X = ZeroPadding1D(padding=3)(X_in)
    X = Conv1D(filters=64, kernel_size=7, strides=2, padding='valid')(X)
    X = ResidualDenseBlock(X, 64, 128, 64, 8, 8, strides=2)
    X = ResidualDenseBlock(X, 128, 256, 128, 16, 8, strides=2)
    X = ResidualDenseBlock(X, 256, 512, 256, 32, 8, strides=2)
    X = ResidualDenseBlock(X, 512, 1024, 512, 64, 8, strides=2)
    X = AveragePooling1D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dense(ndim, activation='sigmoid')(X)
    model = Model(X_in, X)
    return model


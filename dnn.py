# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:34:48 2016

@author: rflamary
"""

import numpy as np
import scipy as sp

np.random.seed(seed=42)

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, normalization
from keras.layers import Dropout,Flatten, Reshape, concatenate
from keras.layers import Convolution2D, MaxPooling2D,UpSampling2D, Merge, merge
from keras.utils import np_utils
from keras.layers import Input, Lambda
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint,EarlyStopping, LearningRateScheduler
from keras.models import model_from_json
from keras.engine.topology import Layer
#from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras import objectives

import time
__time_tic_toc=time.time()

def tic():
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    t=time.time()
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    t=time.time()
    return t-__time_tic_toc


def save_model(model,fname='mymodel'):
    model.save_weights(fname+'.h5',overwrite=True)
    open(fname+'.json', 'w').write(model.to_json())

def load_model(fname):
    model = model_from_json(open(fname+'.json').read())
    model.load_weights(fname+'.h5')
    return model



class Select(Layer):
    def __init__(self, sel, **kwargs):
        self.sel = sel
        self.output_dim=sel[1]-sel[0]
        super(Select, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_dim = input_shape[1]
        #self.trainable_weights = []
        pass

    def call(self, x, mask=None):
        return x[self.sel[0]:self.sel[1]]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
        

from tensorflow.keras import layers

X=[
   [[1,2,3],[4,5,6],   [7,8,9]],
   [[4,5,6],[7,8,9],   [10,11,12]],
   [[7,8,9],[10,11,12],[13,14,15]],
   ...
  ] 
Y = [
     [4],
     [7],
     [10],
     ...
    ]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit([X],[Y],num_epochs=300,validation_split=0.2)

import json
import os
import pickle
import random
import sys
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, RepeatVector, Reshape, TimeDistributed

def main():
    # tf.compat.v1.enable_v2_behavior()

    # create the dataset
    datasetD = make_dataset()

    train_x = datasetD['train']['x']
    train_y = datasetD['train']['y']
    # test_x  = datasetD['test']['x']
    # test_y  = datasetD['test']['y']
    val_x   = datasetD['val']['x']
    val_y   = datasetD['val']['y']

    # create the model
    model = create_model()
    model.compile(optimizer='adam',
                  loss='mse')
    
    print(model.summary())

    history = model.fit([train_x],[train_y],
                        batch_size=32,
                        epochs=300, 
                        validation_data=([val_x],[val_y]),
                        validation_freq=1)

def make_dataset():
    input_window_samps  = 50
    num_signals         = 1
    output_window_samps = 3
    returnD = {}

    returnD['train'] = {}
    returnD['train']['x'] = []
    returnD['train']['y'] = []
    
    for i in range(10000):
        returnD['train']['x'].append(np.arange(i,i+input_window_samps))
        returnD['train']['y'].append(np.expand_dims(np.arange(i+input_window_samps,i+input_window_samps+output_window_samps),axis=1))
    
    returnD['val'] = {}
    returnD['val']['x'] = []
    returnD['val']['y'] = []
    
    for i in range(10000,20000):
        returnD['val']['x'].append(np.arange(i,i+input_window_samps))
        returnD['val']['y'].append(np.expand_dims(np.arange(i+input_window_samps,i+input_window_samps+output_window_samps),axis=1))
    
    returnD['test'] = {}
    returnD['test']['x'] = []
    returnD['test']['y'] = []
    
    for i in range(20000,30000):
        returnD['test']['x'].append(np.arange(i,i+input_window_samps))
        returnD['test']['y'].append(np.expand_dims(np.arange(i+input_window_samps,i+input_window_samps+output_window_samps),axis=1))

    return returnD

def create_model():
    # from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ "Multiple Parallel Input and Multi-Step Output" example 
    input_window_samps  = 50
    num_signals         = 1
    output_window_samps = 3
    units0 = 10
    units1 = 10

    input = Input(shape=(input_window_samps*num_signals,))
    x = Reshape((input_window_samps,num_signals))(input)
    x = LSTM(units0,activation='relu')(x)
    x = RepeatVector(output_window_samps)(x)
    x = LSTM(units1,activation='relu',return_sequences=True)(x)
    x = TimeDistributed(Dense(num_signals))(x)

    model = Model(inputs=input,outputs=x)
    return model
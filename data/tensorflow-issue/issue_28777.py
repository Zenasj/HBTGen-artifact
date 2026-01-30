import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

class CustomModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(CustomModel, self).__init__(**kwargs)
    self.conv1   = Conv2D(32, (3, 3), padding='same')
    self.conv2   = Conv2D(64, (3, 3), padding='same')
    self.pool    = MaxPooling2D(pool_size=(2, 2))
    self.bn      = BatchNormalization()
    self.relu    = Activation("relu")
    self.softmax = Activation("softmax")
    self.drop1   = Dropout(0.25)
    self.drop2   = Dropout(0.5)
    self.dense1  = Dense(512)
    self.dense2  = Dense(10)
    self.flat    = Flatten()
    
  
  
  def call(self, inputs, train):
    z = self.conv1(inputs)
    z = self.bn(z, training=train)
    z = self.relu(z)
    
    z = self.conv1(z)
    z = self.bn(z, training=train)
    z = self.relu(z)
    z = self.pool(z)
    z = self.drop1(z, training=train)
    
    z = self.conv2(z)
    z = self.bn(z, training=train)
    z = self.relu(z)
    
    z = self.conv2(z)
    z = self.bn(z, training=train)
    z = self.relu(z)
    z = self.pool(z)
    z = self.drop1(z, training=train)
    
    z = self.flat(z)
    z = self.dense1(z)
    z = self.relu(z)
    z = self.drop2(z, training=train)
    z = self.dense2(z)
    z = self.softmax(z)
    
    return z

import os
import glob
import h5py
import math
import shutil


import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation
from tensorflow.keras.layers import Input, Flatten, SeparableConv2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as K
import tensorflow as tf

class CustomModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(CustomModel, self).__init__(**kwargs)
    self.conv1   = Conv2D(32, (3, 3), padding='same')
    self.conv2   = Conv2D(64, (3, 3), padding='same')
    self.pool    = MaxPooling2D(pool_size=(2, 2))
    self.bn      = BatchNormalization()
    self.relu    = Activation("relu")
    self.softmax = Activation("softmax")
    self.drop1   = Dropout(0.25)
    self.drop2   = Dropout(0.5)
    self.dense1  = Dense(512)
    self.dense2  = Dense(10)
    self.flat    = Flatten()
    
  
  
  def call(self, inputs, train):
    z = self.conv1(inputs)
    z = self.bn(z, training=train)
    z = self.relu(z)
    
    z = self.conv1(z)
    z = self.bn(z, training=train)
    z = self.relu(z)
    z = self.pool(z)
    z = self.drop1(z, training=train)
    
    z = self.conv2(z)
    z = self.bn(z, training=train)
    z = self.relu(z)
    
    z = self.conv2(z)
    z = self.bn(z, training=train)
    z = self.relu(z)
    z = self.pool(z)
    z = self.drop1(z, training=train)
    
    z = self.flat(z)
    z = self.dense1(z)
    z = self.relu(z)
    z = self.drop2(z, training=train)
    z = self.dense2(z)
    z = self.softmax(z)
    
    return z

model =CustomModel()
random_input = np.random.rand(32,32, 3).astype(np.float32)
random_input = np.expand_dims(random_input, axis=0)
preds = model(random_input, train=False)
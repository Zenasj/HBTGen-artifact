import random
from tensorflow.keras import layers
from tensorflow.keras import models

#Importing Libraries
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Input,Conv2D

#Define network

def net():
  inp = Input(shape=(224,224,3))
  x = Conv2D(64,kernel_size =(3,3),padding='same',activation='relu',strides=2)(inp)
  x = Conv2D(128,kernel_size =(3,3),padding='same',activation='relu',strides=2)(x)
  x = Conv2D(256,kernel_size =(3,3),padding='same',activation='relu',strides=1)(x)
  x = Conv2D(256,kernel_size =(3,3),padding='same',activation='relu',strides=2)(x)
  x = Conv2D(256,kernel_size =(3,3),padding='same',activation='relu',strides=2)(x)
  x = Conv2D(512,kernel_size =(3,3),padding='same',activation='relu',strides=2)(x)
  x = Conv2D(512,kernel_size =(3,3),padding='same',activation='relu',strides=2)(x)
  x = GlobalAveragePooling2D()(x)
  out = Dense(10)(x)
  model = Model(inputs = inp,outputs =out)
  return model

#TPU_init
resolver = tf.contrib.cluster_resolver.TPUClusterResolver()
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = net()
  model.compile(loss = 'categorical_crossentropy',optimizer ='adam')

#Data load
x = np.ones((224,224,3),dtype=np.float32)
n= np.random.randint(0,9)
y= np.zeros((10,),dtype=np.float32)
data = []
labels = []
for i in range(1024):
  data.append(x)
  labels.append(y)

data = np.array(data)
labels = np.array(labels)

#Fit function (Really slow. Should do this 100x faster)
for i in range(1000):
  time1 = time.time()
  model.fit(data,labels,batch_size = 1024,epochs=1)
  time2 = time.time()
  print(time2-time1)

fit_generator

fit

fit
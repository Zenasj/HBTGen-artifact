import random

import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

train_data = np.random.uniform(-1,1,(20,20))

inputs = krs.Input((20,1))

x = inputs

x = krs.layers.Conv1D(1,3,strides = 1,padding='same',dilation_rate=2,activation='relu')(x)
x = krs.layers.Flatten()(x)
x = krs.layers.Dense(10,activation='relu')(x)
x = krs.layers.Dense(2,activation='relu')(x)
x = krs.layers.Dense(10,activation='relu')(x)
x = krs.layers.Dense(20,activation='relu')(x)
x = krs.layers.Reshape(target_shape=(20,1))(x)
x = krs.layers.Conv1DTranspose(1,3,strides=1,dilation_rate=2,padding='same',activation='relu',output_padding=0)(x)
output = krs.layers.Flatten()(x)

model = krs.Model(inputs,output,name='test')

model.compile(optimizer='adam',loss='MSE')

model.summary()

model.fit(train_data,train_data)
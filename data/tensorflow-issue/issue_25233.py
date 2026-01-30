import random
from tensorflow import keras
from tensorflow.keras import models

import numpy as np
train_x = np.random.random((1000, 32))
train_y = np.random.random((1000, 10))

import tensorflow as tf
from tensorflow.keras import layers
inputs = tf.keras.Input(shape=(32,)) 
x = layers.Dense(60, activation='relu')(inputs)
x = layers.Dense(30, activation='relu')(x)
predictions = layers.Dense(10)(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

import keras.backend as K
def scheduler(epoch):
    lr = K.get_value(model.optimizer.lr)
    print("lr:{}".format(lr * 1))
    return K.get_value(model.optimizer.lr)
 
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_x, train_y, batch_size=16, epochs=10,callbacks=[reduce_lr])
b = history.history['lr']

import numpy as np
train_x = np.random.random((1000, 32))
train_y = np.random.random((1000, 10))

from keras.models import Model
from keras.layers import Input, Dense
inputs = Input(shape=(32,)) 
x = Dense(60, activation='relu')(inputs)
x = Dense(30, activation='relu')(x)
predictions = Dense(10)(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mse',
              metrics=['acc'])

import keras.backend as K
from keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    lr = K.get_value(model.optimizer.lr)
    print("lr:{}".format(lr * 1))
    return K.get_value(model.optimizer.lr)
 
reduce_lr = LearningRateScheduler(scheduler)
history = model.fit(train_x, train_y, batch_size=16, epochs=10,callbacks=[reduce_lr])
b = history.history['lr']
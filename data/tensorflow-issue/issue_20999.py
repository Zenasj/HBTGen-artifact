import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay(epoch):
  initial_rate = 1e-3
  factor = int(epoch / 5)
  lr = initial_rate / (10 ** factor)
  return lr

lr_schedule = LearningRateScheduler(step_decay)

input1 = Input(shape=(10,), name="input")
out = Dense(5, activation="relu")(input1)
model = Model(inputs=input1, outputs=out)
model.compile(optimizer= tf.train.AdamOptimizer(1e-3), loss='mse')

np.random.seed(0)
X = np.random.random((20, 10)).astype(np.float32)
Y = np.random.random((20, 5)).astype(np.float32)

model.fit(x=X, y=Y, batch_size=1, epochs=10, callbacks=[lr_schedule])
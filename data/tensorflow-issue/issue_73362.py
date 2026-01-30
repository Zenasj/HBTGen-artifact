import random
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
import numpy as np

x_train = np.random.rand(10000, 10)
y_train = np.random.rand(10000, 1)
w_train = np.random.rand(10000, 1)

def run(with_sample_weights=True):
  model = keras.Sequential()
  model.add(keras.layers.Dense(64, input_dim=x_train.shape[1], activation='relu')) 
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  if with_sample_weights:
    model.fit(x_train, y_train, sample_weight=w_train)
  else:
    model.fit(x_train, y_train)

import time

start = time.time()
for i in range(10):
  run(True)
end = time.time()
print(end - start)

start = time.time()
for i in range(10):
  run(False)
end = time.time()
print(end - start)
import random
from tensorflow import keras
from tensorflow.keras import optimizers

from time import time

import torch
import tensorflow as tf
from tensorflow.keras import layers
from predictor_brain import TFBrain

import numpy as np
data = np.random.random((10000, 32))
labels = np.random.random((10000, 10))

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

a = time()
hist = model.fit(data, 
    labels,
    epochs=10, 
    batch_size=32
    )
b = time()
print(f'time {b-a} seconds')

from time import time

import tensorflow as tf
from tensorflow.keras import layers

import torch

import numpy as np
data = np.random.random((10000, 32))
labels = np.random.random((10000, 10))

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

a = time()
hist = model.fit(data, 
    labels,
    epochs=10, 
    batch_size=32
    )
b = time()
print(f'time {b-a} seconds')
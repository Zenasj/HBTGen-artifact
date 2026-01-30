from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()

model.add(layers.Input(2, name='INPUT'))
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(8, name='layer1', input_shape=(2,), activation='tanh'))
# Add another:
model.add(layers.Dense(4, name='layer2', input_shape=(8,), activation='sigmoid'))
# Add an output layer with 1 output unit:
model.add(layers.Dense(1, name='outputlayer', input_shape=(4,), activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.summary()

print(model.inputs[0].name)
import numpy as np

data = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
labels = np.array([
    [0],
    [1],
    [1],
    [0]
])

model.fit(data, labels, epochs=500)
tf.keras.models.save_model(model, 'model')

for layer,layer_loaded in zip(model.layers,loaded_model.layers):
  print('From Original Model: ', layer.name, ' From loaded_model: ',layer_loaded.name)
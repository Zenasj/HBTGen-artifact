import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np


input_size = 30
output_size = 10

inp = keras.layers.Input((input_size,))
mask = keras.layers.Input((output_size,), dtype=tf.bool)
x = keras.layers.Dense(output_size)(inp)
x = x * tf.cast(mask, dtype=tf.float32)

model = keras.models.Model(inputs=[inp, mask], outputs=x)
model.compile(loss='mean_squared_error', optimizer='sgd')

print('Model inputs: {}'.format(model.inputs)) # Prints: "Model inputs: [<tf.Tensor 'input_1:0' shape=(?, 30) dtype=float32>, <tf.Tensor 'input_2:0' shape=(?, 10) dtype=bool>]"

batch_size = 20
x_train = np.random.rand(batch_size, input_size)
y_train = np.random.rand(batch_size, output_size)
y_train_mask = np.random.rand(batch_size, output_size)
y_train_mask = y_train_mask > .5

model.fit([x_train, y_train_mask], y_train)

checkpoint = './model.h5'
model.save(checkpoint)

model = keras.models.load_model(checkpoint)
print('Model inputs: {}'.format(model.inputs))
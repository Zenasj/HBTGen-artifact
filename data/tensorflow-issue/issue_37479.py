from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


batch_size = 100
dim_input = 100
dim_output = 1
num_iterations = 100 # will consume approx. 5 GB RAM when set to 1000


class CustomMask(tf.keras.layers.Layer):
  def __init__(self):
    super(CustomMask, self).__init__()

  def compute_mask(self, inputs, mask=None):
    batch_size = inputs.shape[0]

    batch_maxes = tf.keras.backend.max(inputs, axis=1)

    for batch in range(batch_size):
      for i in range(num_iterations):
        max = tf.keras.backend.max(batch_maxes[batch])

    return None

  def call(self, inputs, mask=None):
    return inputs


model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(batch_input_shape=(batch_size, dim_input)))

model.add(CustomMask())

model.add(tf.keras.layers.Dense(dim_output))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training_input = np.zeros([batch_size, dim_input])
training_output = np.zeros([batch_size, dim_output])

model.fit(training_input, training_output, batch_size=batch_size)
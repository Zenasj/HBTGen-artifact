import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, Bidirectional, LSTM

class CustomModel(keras.Model):
  def __init__(self, hidden_units):
    super(CustomModel, self).__init__()
    self.lstm = Bidirectional(ConvLSTM2D(filters=16, kernel_size=(1, 1), return_sequences=True, return_state=True))
    self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

  def call(self, inputs, training=None, mask=None):
    x = inputs
    x, _, _, _, _ = self.lstm(x)
    for layer in self.dense_layers:
      x = layer(x)
    return x

model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 10, 10, 10, 5))
outputs=model.predict(input_arr)
model.save('my_model')

# Delete the custom-defined model class to ensure that the loader does not have
# access to it.
del CustomModel

loaded = keras.models.load_model('my_model')
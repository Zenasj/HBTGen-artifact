import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

Python


class Model(tf.keras.Model):
  def __init__(self):
    self.conv = tf.keras.layers.Conv2D(
        128,
        activation='relu')

  def call(self, x):
    return self.conv(x)

x = tf.convert_to_tensor(np.random.random(1, 8, 8))
model = Model()
model(x)
model.save('/tmp/model')

### Relevant log output
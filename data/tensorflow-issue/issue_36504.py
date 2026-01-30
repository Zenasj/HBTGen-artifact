import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

input_layer = tf.keras.layers.Input([32, 32, 3])
model = tf.keras.models.Sequential(
    [input_layer, Flatten(), tf.keras.layers.Dense(10)])

a = np.ones([1, 32, 32, 3], dtype=np.float32)
print(model(a))

class KerasModel(tf.keras.models.Model):

  def __init__(self):
    super(KerasModel, self).__init__()
    self.flatten = Flatten()
    self.dense = tf.keras.layers.Dense(10)

  def call(self, input_layer):
    x = input_layer
    x = self.flatten(x)
    x = self.dense(x)
    return x

model = KerasModel()

a = np.ones([1, 32, 32, 3], dtype=np.float32)
print(model(a))

class KerasModel(tf.keras.models.Model):

  def __init__(self):
    super(KerasModel, self).__init__()
    self.flatten = Flatten()
    self.dense = tf.keras.layers.Dense(10)

  @tf.function(experimental_compile=True)
  def call(self, input_layer):
    x = input_layer
    x = self.flatten(x)
    x = self.dense(x)
    return x

model = KerasModel()

a = np.ones([1, 32, 32, 3], dtype=np.float32)
print(model(a))

model = tf.keras.Model(inputs, outputs)

@tf.function
def train_step(x, y):
  y_pred = model(x)
  ...

class ModelTest(tf.keras.layers.Layer):

  def __init__(self):
    super(ModelTest, self).__init__()
    self.flatten = Flatten()
    self.dense = tf.keras.layers.Dense(10)

  @tf.function
  def call(self, input_layer):
    x = input_layer
    x = self.flatten(x)
    x = self.dense(x)
    return x

input_layer = tf.keras.layers.Input([32, 32, 3])
output_layer = ModelTest()(input_layer)
model = tf.keras.Model(input_layer, output_layer)
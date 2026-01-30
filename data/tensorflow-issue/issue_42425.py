from tensorflow import keras
from tensorflow.keras import layers

loaded = tf.saved_model.load(path)

class LayerFromSavedModel(tf.keras.layers.Layer):
  def __init__(self):
    super(LayerFromSavedModel, self).__init__()
    self.vars = loaded.variables
  def call(self, inputs):
    return loaded.signatures['serving_default'](inputs)

input = tf.keras.Input(...)
model = tf.keras.Model(input, LayerFromSavedModel()(input))
model.save('saved_model')

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

loaded = tf.saved_model.load(r"C:\Users\my_user\Documents\out\2\OUTPUT\model")

class LayerFromSavedModel(tf.keras.layers.Layer):
  def __init__(self):
    super(LayerFromSavedModel, self).__init__()
    self.vars = loaded.variables
  def call(self, inputs):
    return loaded.signatures['serving_default'](inputs)

input = tf.keras.Input(...)
model = tf.keras.Model(input, LayerFromSavedModel()(input))
model.save('saved_model')
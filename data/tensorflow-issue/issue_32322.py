import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

class RandomMaskingNoAlter(tf.keras.layers.Layer):

  def call(self, inputs):
    return inputs

  def compute_mask(self, inputs, mask=None):
    print('no alter executed')
    if mask is None:
      return None
    random_mask = tf.cast(tf.random.uniform(tf.shape(mask), 0, 1, dtype=tf.int32), tf.bool)
    return tf.math.logical_and(random_mask, mask)

class RandomMaskingAlter(tf.keras.layers.Layer):
    
  def call(self, inputs):
    return inputs + 0

  def compute_mask(self, inputs, mask=None):
    print('alter executed')
    if mask is None:
      return None
    random_mask = tf.cast(tf.random.uniform(tf.shape(mask), 0, 1, dtype=tf.int32), tf.bool)
    return tf.math.logical_and(random_mask, mask)

x = np.array([[1, 4, 2, 2, 0, 0], [1, 1, 1, 0, 0, 0], [3, 2, 2, 3, 4, 1]], dtype='i4')
y = tf.keras.layers.Embedding(5, 5, mask_zero=True)(x)

z1 = RandomMaskingNoAlter()(x)
z2 = RandomMaskingNoAlter()(y)
z3 = RandomMaskingAlter()(x)
z4 = RandomMaskingAlter()(y)

class RandomMaskingNoAlter(tf.keras.layers.Layer):

  def call(self, inputs):
    return inputs

class RandomMaskingNoAlter(tf.keras.layers.Layer):

  def call(self, inputs):
    return inputs+0
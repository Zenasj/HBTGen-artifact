import random
from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import numpy as np
input = np.random.uniform(size=(6, 40, 40, 1)).astype(np.float32)

ds = tf.data.Dataset.from_tensor_slices(input).batch(2)

def _print(t):
  tf.print(t)
  return t

iterator = iter(ds)
inp = tf.keras.Input((None, None, 1))
out = tf.keras.layers.Lambda(_print)(inp)
out = tf.keras.layers.Conv2D(5, 3)(out)
model = tf.keras.Model(inputs=inp, outputs=out)
model(next(iterator))
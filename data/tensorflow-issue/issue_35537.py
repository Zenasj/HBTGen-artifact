import math
import random
from tensorflow import keras

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations


class SFM1(tf.keras.Model):
    def __init__(self):
        super(SFM1, self).__init__()
        self.output_layer = layers.Activation('softmax', dtype='float32')
    
    def call(self, inputs):
        return self.output_layer(inputs)

class SFM2(tf.keras.Model):
    def __init__(self):
        super(SFM2, self).__init__()
        
    
    def call(self, inputs):
        return activations.softmax(inputs, axis=1)
    
x = tf.random.uniform((2, 4, 64, 64, 64), dtype=tf.float32)
sfm1 = SFM1()
y1 = sfm1(x)

sfm2 = SFM2()
y2= sfm2(x)

tf.math.equal(y1, y2)

tf.dtypes.cast(activations.softmax(inputs, axis=1), dtype=tf.float32)

activations.softmax(tf.dtypes.cast(inputs, dtype=tf.float32), axis=1)

layer = tf.keras.layers.Softmax(axis=1, dtype=tf.float32)
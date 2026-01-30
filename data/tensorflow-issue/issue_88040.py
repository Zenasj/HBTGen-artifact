from tensorflow import keras

import os
import tensorflow
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class FFTInverseModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

    def call(self, x):
        inv = tf.linalg.inv(x)
        return inv


model = FFTInverseModel()

input_shape = (2, 2)

x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.complex64, shape=input_shape)  # tf.complex64 is the trigger condition

inputs = [x]

model(*inputs)
print("succeed on eager")


class FFTInverseModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        inv = tf.linalg.inv(x)
        return inv


model = FFTInverseModel()
model(*inputs)
print("succeed on XLA")
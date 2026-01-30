import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class OpOrSkip(tf.keras.layers.Layer):
    def __init__(self, op):
        super().__init__()
        self.op = op
        
    def call(self, x):
        rnd = tf.random.uniform(())
        if rnd < 0.5:
            return self.op(x)
        else:
            return x

def skip_conv(s):
    x = tf.keras.layers.Conv2D(3, 3, padding='same')(s)
    return x + s #tf.identity(s, name='s')

def func_as_model(func, shape):
    inp = tf.keras.Input(shape)
    out = func(inp)
    return tf.keras.Model(inputs=inp, outputs=out)

inputs = tf.keras.Input((32, 32, 3))
SkipConv = func_as_model(skip_conv, [32, 32, 3])

x = OpOrSkip(SkipConv)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=x)
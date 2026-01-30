from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class RNNCellWithConstants(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.state_size = 5
        super(RNNCellWithConstants, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        self.built = True

    def call(self, inputs, states, constants):
        print(inputs, states, constants)
        return inputs, [inputs]


# Test basic case.
x = tf.keras.Input((None, 5))
c = tf.keras.Input((3,))
cell = RNNCellWithConstants()
layer = tf.keras.layers.RNN(cell)
y = layer(x, constants=c) # Works as expected.

# Test basic case.
x = tf.zeros([3, 3, 5], dtype=tf.float32)
c = tf.zeros([3, 3], dtype=tf.float32)
cell = RNNCellWithConstants()
layer = tf.keras.layers.RNN(cell)
y = layer(x, constants=c) # Crash with the following error
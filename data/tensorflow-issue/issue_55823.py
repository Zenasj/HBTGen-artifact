import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class BasicLayer(tf.keras.layers.Layer):
  def build(self, input_shape):
    self.w = self.add_weight(name='w', shape=(3,),
                             initializer=tf.keras.initializers.Zeros())

  def call(self, inputs):
    return inputs * self.w

# Input/output data.
x = tf.constant([[1. + 2.j, 2. + 3.j, 3. + 4.j]])
y = x

# Create and compile model caling this layer.
input = tf.keras.Input(shape=(3,), dtype='complex64')
layer = BasicLayer(dtype='complex64')
model = tf.keras.Model(input, layer(input))
model.compile('rmsprop', 'mse')
model.train_on_batch(x, y)

import tensorflow as tf

tf.debugging.set_log_device_placement(True)

x = tf.constant(1.0 + 2.0j, dtype=tf.complex64)
y = tf.constant(1.0 + 2.0j, dtype=tf.complex64)
x = tf.math.sqrt(tf.math.square(x) + tf.math.square(y))
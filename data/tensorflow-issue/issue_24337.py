import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inpt = tf.keras.layers.Input(shape=(257, 257, 3))
out = tf.keras.layers.DepthwiseConv2D(
    kernel_size=64, dilation_rate=2, add_bias=False
)
model = tf.keras.Model(inpt, [out])

inpt = tf.keras.layers.Input(shape=(257, 257, 3))
out = tf.keras.layers.DepthwiseConv2D(
    kernel_size=64, dilation_rate=2, add_bias=False
)
out = tf.keras.layers.Lambda(lambda x: x + 0.0)
model = tf.keras.Model(inpt, [out])
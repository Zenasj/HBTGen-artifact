import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
data_format = "channels_first"
return_sequences = False
filters = 2
kernel_size = [0, 1]
padding = "valid"
layer = tf.keras.layers.ConvLSTM2D(data_format=data_format,return_sequences=return_sequences,filters=filters,kernel_size=kernel_size,padding=padding,)
x = tf.keras.Input(shape = (2, 2, 5, 5))
y = layer(x)
model = tf.keras.Model(x, y)
input = tf.random.uniform((2, 2, 2, 5, 5), dtype=tf.float32)
res = model(input)
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

input = tf.keras.Input(shape=())
output = tf.keras.layers.Lambda(lambda x: x, dynamic=True)(input)
tf.keras.Model(inputs=input, outputs=output)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[7]))
layer = tf.keras.layers.Dense(units=20, activation='relu')
layer1 = tf.keras.layers.Dense(units=20, activation='relu')
model.add(layer)
model.add(layer1)
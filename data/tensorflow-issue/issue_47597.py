from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# Make a model with 2 layers
layer1 = tf.keras.layers.Dense(3, activation="relu")
layer2 = tf.keras.layers.Dense(3, activation="sigmoid")
model = tf.keras.Sequential([tf.keras.Input(shape=(3,)), layer1, layer2])

# Freeze the first layer
layer1.trainable = False

model.summary()

import tensorflow as tf

# Make a model with 2 layers
layer1 = tf.keras.layers.Dense(3, activation="relu")
layer2 = tf.keras.layers.Dense(3, activation="sigmoid")
model = tf.keras.Sequential([tf.keras.Input(shape=(3,)), layer1, layer2])

# Freeze the first layer
layer1.bias.trainable = False

model.summary()
from tensorflow import keras
from tensorflow.keras import layers

import math
import numpy as np
import tensorflow as tf

# simple dataset with zeros
batch_size = 32
features = np.zeros((10000, 60, 2))
labels = np.zeros((10000, 1))
train_data = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
train_steps = int(math.ceil(features.shape[0] / batch_size))

# simple model with Dense layers
inputs = tf.keras.Input(shape=(features[0].shape[0], features[0].shape[1]))
x = tf.keras.layers.Dense(32, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1, activation="relu")(x)
model = tf.keras.Model(inputs, outputs, name="example_model")

# model fitting
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(train_data, epochs=100, steps_per_epoch=train_steps)
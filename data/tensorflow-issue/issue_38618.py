import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
data = tf.random.uniform(shape=(1000, 784),  maxval=15)
labels = tf.random.uniform(shape=(1000,10), maxval = 1, dtype=tf.int32)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation = "relu"),
  tf.keras.layers.Dense(10, activation = "softmax")
])
model.compile(loss = "binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(data, labels, verbose=1)
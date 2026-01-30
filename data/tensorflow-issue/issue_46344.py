import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_shape=(5,)))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer="Adam", loss="binary_crossentropy")
x = np.random.rand(4,5)
y = np.random.randint(0, 2, (4,))
model.fit(x, y, epochs=10, callbacks=[tf.keras.callbacks.BaseLogger()])

x = np.random.rand(4,5)
y = np.random.randint(0, 2, (4,))
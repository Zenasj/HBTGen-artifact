import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

data = np.random.randn(1000, 10)
targets = np.random.randn(1000, 1)
ds = tf.data.Dataset.from_tensor_slices(data).batch(32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim=10))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer="rmsprop", loss="mse")
model.fit(ds, steps_per_epoch=10, validation_data=(data, targets), batch_size=32)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION, flush=True)
print(tf.config.list_physical_devices(), flush=True)


month_input_layer = tf.keras.layers.Input(shape=(1,), name="month", dtype="int64")
month_encoded_layer = tf.compat.v1.keras.layers.experimental.preprocessing.CategoryEncoding(output_mode="binary")
dense_layer = tf.keras.layers.Dense(1,activation="relu")

month_encoded_layer.adapt(np.arange(1,13))
output = dense_layer(month_encoded_layer(month_input_layer))

model = tf.keras.models.Model(month_input_layer, output)

model.summary()
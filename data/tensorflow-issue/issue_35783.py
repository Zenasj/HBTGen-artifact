from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

fetches = [lambda: whatever_I_write_here_is_ignored]

var = tf.Variable([[3.0]])
model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(1,))])
model.compile(loss="mse", optimizer="adam")
model._function_kwargs = {"fetches": fetches, "should_fail": "ignored_as_well"}

model.fit([[7.0]], [[9.0]], epochs=2)

import keras
import tensorflow as tf

fetches = [lambda: whatever_I_write_here_is_ignored]

var = tf.Variable([[3.0]])
model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(1,))])
model.compile(loss="mse", optimizer="adam")
model._function_kwargs = {"fetches": fetches, "should_fail": "ignored_as_well"}

model.fit([[7.0]], [[9.0]], epochs=2)
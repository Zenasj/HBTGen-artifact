from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

dtype = 'float64'

some_input = keras.Input(shape=(1), dtype=dtype)
some_value = keras.layers.Dense(4, activation="relu", dtype=dtype)(some_input)
some_value = keras.layers.Dense(2, dtype=dtype)(some_value)

other_value = tf.constant([0.1], dtype=dtype)

mask = tf.equal([0, 1], 1)
replaced_value = tf.where(mask, x=other_value, y=some_value)

model = tf.keras.Model(inputs=[some_input], outputs=[replaced_value])
model.save('my_test_model')

loaded_model = tf.keras.models.load_model('my_test_model')
import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

dummy_input = tf.random.normal((1024, 1000))
dummy_output = tf.random.normal((1024, 10))
dummy_output_weights = tf.random.uniform(minval=0, maxval=1, shape=(1024, 10))

print(dummy_input.shape)
print(dummy_output.shape)
print(dummy_output_weights.shape)

dummy_model = tf.keras.Sequential()
dummy_model.add(tf.keras.Input(shape=(1000,)))
dummy_model.add(tf.keras.layers.Dense(32, activation='relu'))
dummy_model.add(tf.keras.layers.Dense(10, activation='linear'))

dummy_model.compile(optimizer='sgd', loss='mse', sample_weight_mode='temporal')
dummy_model.summary()

dummy_model.fit(x=dummy_input, y=dummy_output, batch_size=32, epochs=10, sample_weight=dummy_output_weights)
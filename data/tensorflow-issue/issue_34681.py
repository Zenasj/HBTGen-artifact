import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


batch_size = 16
num_batches = 4
num_timesteps = 100
num_features = 40
num_targets = 2

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(2, input_shape=(None, num_features), return_sequences=True))
model.compile(optimizer='adam', loss='mse')
model.summary()


X = tf.random.normal((batch_size * num_batches, num_timesteps, num_features))
y = tf.random.normal((batch_size * num_batches, num_timesteps, num_targets))

model.fit(X, y)
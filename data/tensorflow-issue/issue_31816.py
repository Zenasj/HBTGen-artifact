import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

def train_generator():
    while True:
        sequence_length = np.random.randint(10, 100)
        x_train = np.random.random((1000, sequence_length, 5))
        y_train = np.random.random((1000, sequence_length, 2))

        yield x_train, y_train

# Works as intended
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None, 5)))
model.add(tf.keras.layers.LSTM(8, return_sequences=True))
model.add(tf.keras.layers.Dense(2))
model.compile(optimizer="adam", loss="mse")
model.fit_generator(train_generator(), steps_per_epoch=2, epochs=2, verbose=1)

# Throws an exception
class LSTMModel(tf.keras.Model):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self._lstm_0 = tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None, 5)) 
        self._lstm_1 = tf.keras.layers.LSTM(8, return_sequences=True)
        self._dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        output = self._lstm_0(inputs)
        output = self._lstm_1(output)
        output = self._dense(output)

        return output

model = LSTMModel()
model.compile(optimizer="adam", loss="mse")
model.fit_generator(train_generator(), steps_per_epoch=2, epochs=2, verbose=1)
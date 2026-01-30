from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
input = keras.layers.Input(shape=[12, 100], batch_size=1)
x = keras.layers.Dense(20, activation='relu')(input)
x = keras.layers.BatchNormalization()(x)
fc1 = keras.layers.Dense(10, activation=None)(x)
gru1 = keras.layers.LSTM(20, return_sequences=True, stateful=True)(fc1)
gru2 = keras.layers.LSTM(20, return_sequences=True, stateful=True)(gru1)
model = keras.Model(inputs=[input], outputs=[fc1, gru2])
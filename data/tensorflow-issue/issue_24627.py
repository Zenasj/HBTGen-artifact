from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

input_data = keras.layers.Input(shape=[8])
hidden1 = keras.layers.Dense(30, activation="relu")(input_data)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_data, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_data], outputs=[output])
model.compile(loss="mean_squared_error", optimizer="sgd")

model.summary()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=["accuracy"], optimizer="SGD")

# Compiling again with the same loss, optimizer and metrics
model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)

model.save("my_model.h5")
loaded = tf.keras.models.load_model("my_model.h5")

# Compiling again with the same loss, optimizer but different metrics
model.compile(loss=model.loss, optimizer=model.optimizer, metrics=["accuracy"])

model.save("my_model.h5")
loaded = tf.keras.models.load_model("my_model.h5")
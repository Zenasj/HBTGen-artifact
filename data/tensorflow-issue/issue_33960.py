from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from urllib.request import urlopen
import numpy as np
import importlib
import os

# here's how data is loaded
dataset = np.loadtxt(raw_data, delimiter=",")

# build model
model = Sequential()
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(32, activation='softmax'))
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

# training (data has been loaded via np.loadtxt)
model.fit(
    training_dataset,
    training_labels,
    epochs=20,
    shuffle=True,
    validation_data=(validation_dataset, validation_labels)
)
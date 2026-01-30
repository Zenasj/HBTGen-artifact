import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D

x = np.random.rand(1000, 401, 17)
y = np.random.choice([0, 1], size=(1000, 301))

model = Sequential()
model.add(Conv1D(filters=320, kernel_size=26, activation='relu', input_shape=(401, x.shape[2])))
model.add(MaxPooling1D(pool_size=13, strides=13))
model.add(Bidirectional(GRU(320, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Flatten())
model.add(Dense(2000, activation="relu"))
model.add(Dense(301, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.summary()

model.fit(x=x, y=y, epochs=1, verbose=1)
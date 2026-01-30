import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

x = np.random.normal(size=10)
y = np.random.normal(size=10)

model = Sequential()
model.add(Dense(1))

reduce_lr = ReduceLROnPlateau(monitor='loss',
                              min_delta=0.01,
                              patience=3,
                              min_lr=0.001,
                              verbose=2)

model.compile(loss='mse', optimizer=SGD(0.01))

history = model.fit(x,
                    y,
                    epochs=25,
                    verbose=2,
                    shuffle=True,
                    batch_size=1,
                    callbacks=[reduce_lr])

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD

x = np.random.normal(size=10)
y = np.random.normal(size=10)

model = Sequential([Dense(1)])


reduce_lr = ReduceLROnPlateau(monitor='loss',
                              min_delta=0.01,
                              patience=3,
                              min_lr=0.001,
                              verbose=2)

model.compile(loss='mse', optimizer=SGD(0.01))

history = model.fit(x,
                    y,
                    epochs=50,
                    verbose=2,
                    shuffle=True,
                    batch_size=1,
                    callbacks=[reduce_lr])
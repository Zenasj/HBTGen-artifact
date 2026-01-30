from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import time

from tensorflow import keras


for _ in range(100):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(120, activation='relu'))
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD())
    time.sleep(0.1)

import time

from tensorflow import keras


for _ in range(100):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(120, activation='relu', input_shape=(10, 10)))
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD())
    time.sleep(0.1)

from pympler import muppy
from pympler import summary

...

all_objects = muppy.get_objects()
occupancy = summary.summarize(all_objects)
summary.print_(occupancy)
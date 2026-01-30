import random
from tensorflow.keras import layers

import tensorflow.keras as keras
import numpy as np

model = keras.Sequential()
model.add(keras.Input(shape=(), batch_size=5))
model.add(keras.layers.Activation('sigmoid'))
model.compile(loss=keras.losses.BinaryCrossentropy())
xs = np.random.normal(size=200)
ys = np.random.randint(0, 2, 200)
cb = keras.callbacks.TensorBoard('tmp')
model.fit(xs, ys, callbacks=[cb], batch_size=5)
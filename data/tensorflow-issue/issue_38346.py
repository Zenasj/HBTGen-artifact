import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
  
x = tf.keras.Input(shape=(None, 12), dtype="float32", name="input")
x_frequency = tf.signal.frame(x, 2, 1, axis=1)
model = tf.keras.Model(inputs=x, outputs=x_frequency, name="test")
model.save("test.h5")

keras_model = tf.keras.models.load_model("test.h5")

import tensorflow as tf
  
x = tf.keras.Input(shape=(), dtype="float32", name="input")
x_reshape = tf.reshape(x, [])
model = tf.keras.Model(inputs=x, outputs=x_reshape, name="test1")
model.save("test1.h5")

keras_model = tf.keras.models.load_model("test1.h5")

DNN_model = Sequential()
DNN_model.add(Dense(units = 1, activation = 'elu'))
DNN_model.add(Dense(units = 10, activation = 'elu'))
DNN_model.add(Dense(units = 1))
#
DNN_model.compile(loss = MeanSquaredError())
#
DNN_model.fit(np.array(x), np.array(y),
              epochs = 25,
              verbose = 1)
#
DNN_model.save('keras_parallel_' + ts() + str(np.random.randint(1e8, 9e8)) + '.hdf5')

np.array(x).reshape(-1, 1)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


def generator():
      while True:
        yield np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)


with strategy.scope():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=(10,), activation="relu"))
  model.compile('Adam', 'mae')
  model.fit(generator(), steps_per_epoch=1000, epochs=10)

import numpy as np
import tensorflow as tf
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


def generator():
    while True:
        yield [np.ones([10, 10], np.float32), np.ones([10, 10], np.float32)], np.ones([10, 1], np.float32)


with strategy.scope():
    inputA = tf.keras.layers.Input(shape=(10,))
    inputB = tf.keras.layers.Input(shape=(10,))

    output = tf.keras.layers.Concatenate()([inputA, inputB])
    output = tf.keras.layers.Dense(1, input_shape=(10,), activation="relu")(output)
    model = tf.keras.models.Model(inputs=[inputA, inputB], outputs=output)
    model.compile('Adam', 'mae')
    model.fit(generator(), steps_per_epoch=1000, epochs=10)
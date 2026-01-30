import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np


class M1(tf.keras.Model):
    def __init__(self, **kwargs):
        super(M1, self).__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(units=1)

    @tf.function
    def call(self, inputs, training=None):
        x, y, z = inputs
        x = tf.math.unsorted_segment_sum(x, tf.squeeze(y), z)
        return self.fc(x)


class M2(tf.keras.Model):
    def __init__(self, **kwargs):
        super(M2, self).__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(units=1)

    @tf.function
    def call(self, inputs, training=None):
        x, y, z = inputs
        x = tf.math.segment_sum(x, tf.squeeze(y))
        return self.fc(x)


def gen():
    for _ in range(1024):
        offset = np.random.randint(1, 10, size=1024)
        y = np.repeat(np.arange(1024), offset)
        z = 1024
        x = np.random.rand(offset.sum(), 32)
        yield (x, y, z), np.random.rand(1024)


ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=((tf.TensorSpec(shape=(None, 32), dtype=tf.float64),
                      tf.TensorSpec(shape=(None, ), dtype=tf.int64),
                      tf.TensorSpec(shape=[], dtype=tf.int32)),
                       tf.TensorSpec(shape=(None, ), dtype=tf.float64))
)



m1, m2 = M1(), M2()
m1.compile(loss=tf.keras.losses.MeanSquaredError(),
           optimizer=tf.keras.optimizers.Adagrad())
m2.compile(loss=tf.keras.losses.MeanSquaredError(),
           optimizer=tf.keras.optimizers.Adagrad())
# tf.math.segment_* is OK
m2.fit(ds, epochs=8)
# tf.math.unsorted_segment_* FAILED
m1.fit(ds, epochs=8)
import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

source = (
    ((tf.constant(np.random.normal(0, 1, (1024, 2)), dtype=tf.float32),
      tf.constant(np.random.normal(0, 1, (1024, 2)), dtype=tf.float32))),
    tf.constant(np.random.randint(0, 2, (1024, 2)), dtype=tf.float32)
)

train = tf.data.Dataset.from_tensor_slices(source).batch(128, drop_remainder=True).repeat(10)
valid = tf.data.Dataset.from_tensor_slices(source).batch(128, drop_remainder=True).repeat(10)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.d = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, training=True, mask=None):
        return self.d(inputs[0] + inputs[1])


m = Model()
m.compile(tf.train.AdamOptimizer(0.001),
          loss=["categorical_crossentropy"])
m.fit(x=train, validation_data=valid, steps_per_epoch=8)

import tensorflow as tf
from collections import ChainMap

tf.enable_eager_execution()


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x1 = inputs['x1']
        x2 = inputs['x2']

        x1 = self.dense1(x1)
        x2 = self.dense2(x2)

        y = self.dense3(x1 + x2)

        return y


model = MyModel()

x1 = tf.random.normal((10, 1), 0, 1)
x2 = tf.random.normal((10, 1), 1, 2)
y = x1 + 2 * x2

ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices({'x1': x1}),
                          tf.data.Dataset.from_tensor_slices({'x2': x2}))) \
    .map(lambda *dicts: dict(ChainMap(*dicts)))

y = tf.data.Dataset.from_tensor_slices(y)

ds = tf.data.Dataset.zip((ds, y)) \
    .batch(5) \
    .repeat()

model.compile(tf.train.AdamOptimizer(), 'mean_squared_error', metrics =['mae'])
model.fit(ds, epochs=1, steps_per_epoch=10, validation_data=ds)
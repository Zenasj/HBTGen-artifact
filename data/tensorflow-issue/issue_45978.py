import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import types

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ProgbarLogger

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

metrics = [tf.keras.metrics.CategoricalAccuracy()]
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=metrics)

np.random.seed(0)
tf.random.set_seed(0)
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

metric_names = [
    m.lower() if isinstance(m, str) else
    m.__name__ if isinstance(m, types.FunctionType) else
    m.name for m in metrics]
callbacks = [ProgbarLogger(
    count_mode='steps',
    stateful_metrics=metric_names)]

result = model.fit(dataset, callbacks=callbacks, epochs=10)

#!/usr/bin/env python
import types

import numpy as np
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


class ReservoirHistogram(tf.metrics.Metric):

    def __init__(self,
                 name='histogram',
                 reservoir_size=300,
                 reservoir_shape=(10,),
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.reservoir: tf.Variable = self.add_weight(
            name='reservior',
            shape=(reservoir_size,) + reservoir_shape,
            initializer='zeros',
            dtype=self.dtype)

        self.current_index: tf.Variable = self.add_weight(
            name='current_index', shape=(), dtype=tf.int32,
            initializer=tf.constant_initializer(0))

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_pred)[0]
        reservoir_size = tf.cast(tf.shape(self.reservoir)[0], tf.int32)
        r_index = self.current_index % reservoir_size
        batch_part = tf.minimum(batch_size, reservoir_size - r_index)
        batch_random = tf.random.uniform(shape=(batch_part,), maxval=1.)
        self.current_index.assign_add(batch_part)
        reservoir_prob = (1. if self.current_index < reservoir_size
                          else tf.cast(reservoir_size / self.current_index,
                                       dtype=tf.float32))
        batch_mask = batch_random <= reservoir_prob
        batch_mask = tf.tile(
            input=tf.expand_dims(tf.cast(batch_mask, tf.float32), axis=1),
            multiples=tf.concat(((1,), tf.shape(self.reservoir)[1:]), axis=0))
        old_values = self.reservoir[r_index: r_index + batch_part]
        batch_values = y_pred[:batch_part]
        # transform values to have a zero-centered distribution
        batch_values = tf.clip_by_value(tf.where(
            batch_values > 0.5, batch_values - 1, batch_values),
            clip_value_min=-0.1, clip_value_max=0.1)
        new_values = old_values * (1 - batch_mask) + batch_values * batch_mask
        new_reservoir = tf.concat(
            (self.reservoir[:r_index],
             new_values,
             self.reservoir[r_index + batch_part:]),
            axis=0)
        self.reservoir.assign(new_reservoir)

    def result(self):
        used_size = tf.minimum(self.current_index, tf.shape(self.reservoir)[0])
        return self.reservoir[:used_size]

    def reset_states(self):
        self.reservoir.assign(tf.zeros(shape=tf.shape(self.reservoir)))
        self.current_index.assign(0)


metrics = [tf.keras.metrics.CategoricalAccuracy(), ReservoirHistogram()]
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=metrics)

np.random.seed(0)
tf.random.set_seed(0)
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

if False:  # switch to True to demonstrate that the hotfix works
    metric_names = [
        m.lower() if isinstance(m, str) else
        m.__name__ if isinstance(m, types.FunctionType) else
        m.name for m in metrics]
    callbacks = [tf.keras.callbacks.ProgbarLogger(
        count_mode='steps',
        stateful_metrics=metric_names)]
    result = model.fit(dataset, callbacks=callbacks, epochs=10)
else:
    result = model.fit(dataset, epochs=10)
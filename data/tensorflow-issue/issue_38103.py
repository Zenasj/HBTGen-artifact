import random
from tensorflow import keras
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.eager import backprop

class TestModel(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.x = layers.Input((10,))
        self.output_ = layers.Dense(1)

    def call(self, inputs):
        return self.output_(inputs)

    @tf.function(
        input_signature=[(tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, 1), dtype=tf.int64))])
    def train_step(self, data):
        tf.print('Flag')
        x, y = data
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({'train_step': self.train_step})
        return config

original_model = TestModel()
original_model.compile(loss='mse', optimizer='sgd')

for i in range(3):
    original_model.train_on_batch(np.random.rand(6, 10), np.arange(6))
    print('Completed train loop (original model)')

save_path = 'tmp'
original_model.save(save_path)

loaded_model = tf.keras.models.load_model(save_path)

for i in range(3):
    loaded_model.train_on_batch(np.random.rand(6, 10), np.arange(6))
    print('Completed train loop (loaded model)')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.eager import backprop


class TestModel(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.x = layers.Input((10,))
        self.output_ = layers.Dense(1)

    def call(self, inputs):
        return self.output_(inputs)

    @tf.function(
        input_signature=[(tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, 1), dtype=tf.int64))])
    def train_step(self, data):
        tf.print('Flag')
        x, y = data
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({'train_step': self.train_step})
        return config


original_model = TestModel()
original_model.compile(loss='mse', optimizer='sgd')

for i in range(3):
    original_model.train_on_batch(np.random.rand(6, 10), np.arange(6))
    print('Completed train loop (original model)')

save_path = 'tmp'
original_model.save(save_path)

loaded_model = tf.keras.models.load_model(save_path)

for i in range(3):
    loaded_model.train_on_batch(np.random.rand(6, 10), np.arange(6))
    print('Completed train loop (loaded model)')

train_step

train_step

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.eager import backprop


class TestModel(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.x = layers.Input((10,))
        self.output_ = layers.Dense(1)

    def call(self, inputs):
        return self.output_(inputs)

    @tf.function(input_signature=[tuple(((
            [tf.TensorSpec((None, 10), tf.float32),
             tf.TensorSpec((None,), tf.int64)])))])
    def train_step(self, data):
        tf.print('Flag')
        x, y = data
        assert isinstance(x, tf.Tensor)
        assert isinstance(y, tf.Tensor)
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({'train_step': tf.function(self.train_step)})
        return config

original_model = TestModel()
original_model.compile(loss='mse', optimizer='sgd')

for i in range(3):
    original_model.train_on_batch(
        np.random.rand(6, 10), np.arange(6))
    print('Completed train loop (original model)')

save_path = 'tmp'
original_model.save(save_path)

loaded_model = tf.keras.models.load_model(save_path, compile=False)
loaded_model.compile(loss='mse', optimizer='sgd')

for i in range(3):
    loaded_model.train_on_batch(np.random.rand(6, 10), np.arange(6))
    print('Completed train loop (loaded model)')
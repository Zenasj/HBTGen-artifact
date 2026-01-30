import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

class MyModel(keras.models.Model):

    def build(self, batch_input_shape):
        self.output_layer = keras.layers.Dense(1)
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        self.add_loss(tf.reduce_mean(self.output_layer(inputs)))
        return self.output_layer(inputs)

model = MyModel()
model.compile(loss="mse", optimizer="nadam")

X = tf.random.uniform((100, 10))
y = tf.random.uniform((100, 1))
history = model.fit(X, y, epochs=2)

import tensorflow as tf
from tensorflow import keras

class MyModel(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, training=None):
        self.add_loss(tf.reduce_mean(self.output_layer(inputs)))
        return self.output_layer(inputs)

model = MyModel()
model.compile(loss="mse", optimizer="nadam")

X = tf.random.uniform((100, 10))
y = tf.random.uniform((100, 1))
history = model.fit(X, y, epochs=2)

def build(self, batch_input_shape):
        super().build(batch_input_shape)

model = MyModel()
model.build((None, 10))
model.compile(loss="mse", optimizer="nadam")
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
        self.add_loss(1.0)
        return self.output_layer(inputs)

model = MyModel()
model.compile(loss="mse", optimizer="nadam")

X = tf.random.uniform((100, 10))
y = tf.random.uniform((100, 1))
history = model.fit(X, y, epochs=2)

class MyModel2(keras.models.Model):
    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        self.add_loss(1.0)
        return tf.reduce_mean(inputs)

model2 = MyModel2()
model2.compile(loss="mse", optimizer="nadam")
history = model2.fit(X, y, epochs=2)
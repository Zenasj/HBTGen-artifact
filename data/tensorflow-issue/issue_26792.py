import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

add_y = Lambda(lambda x: tf.math.add(x,K.variable([0.0], dtype=tf.float32, name='x')))

class Bias(tf.keras.layers.layer):
  def build(self, input_shape):
    self.bias = self.add_weight(shape=(), initializer='zeros', dtype=tf.float32, name='x')

  def call(self, inputs):
    return inputs + self.bias

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

class Bias(keras.layers.Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(1,), initializer='zeros', dtype=tf.float32, name='x')
        super().build(input_shape)

    def call(self, x):
        temp = tf.reduce_mean(x, axis=-1, keepdims=True)
        return temp * 0 + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

x_input = keras.layers.Input(shape=(2,))
V = Bias()(x_input)
model = keras.models.Model(inputs=x_input, outputs=V)
model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam())

X = np.random.randn(100,2)
y = np.random.randn(100, 1) + 1.2
print(model.predict(X).squeeze())
model.fit(X, y=y, epochs=10)
print(model.predict(X).squeeze())
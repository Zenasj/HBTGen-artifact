import random
from tensorflow import keras

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.estimator import model_to_estimator_v2

x_shape = (3,)
n_class = 5
batch_size = 10

x = layers.Input(shape=x_shape, name="x", dtype=tf.int64)

class StupidLayer(layers.Layer):
    def build(self, input_shape):
        self.y = tf.random.poisson(
            lam=10, shape=(batch_size,) + x_shape, dtype=tf.int64
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.cast(inputs * self.y, tf.float32)

y = StupidLayer()(x)

model = tf.keras.Model(inputs=x, outputs=y)

X = np.random.randint(50, size=(batch_size,) + x_shape, dtype="int64")

model.compile(optimizer="sgd", loss="categorical_crossentropy")
model.predict(X)

tf_estimator = model_to_estimator_v2(keras_model=model)
next(tf_estimator.predict(lambda: X))
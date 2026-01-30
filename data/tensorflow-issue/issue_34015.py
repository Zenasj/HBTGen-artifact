import random
from tensorflow import keras
from tensorflow.keras import layers

def build(self, input_shape):
        """Build layer."""
        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer=tf.initializers.identity(),
            trainable=True,
        )

def build(self, input_shape):
        """Build layer."""
        self.w = self.add_weight(
            shape=(input_shape[-1].value, input_shape[-1].value),
            initializer=tf.initializers.identity(),
            trainable=True,
        )

import tensorflow as tf
import numpy as np


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, *args, kernel_initializer=tf.initializers.identity(), **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        """Build layer."""
        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer=self.kernel_initializer,
            trainable=True,
        )

    def call(self, inputs):
        """Apply layer."""
        return tf.matmul(inputs, tf.expand_dims(self.w, 0))


if __name__ == "__main__":
    tf.enable_eager_execution()
    inputs = np.random.normal(size=(1, 10, 3))
    layer = MyLayer()
    outputs = layer(inputs)
    print(outputs.numpy())

import tensorflow as tf
import numpy as np


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, *args, kernel_initializer=tf.initializers.ones(), **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        """Build layer."""
        self.w = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer=self.kernel_initializer,
            trainable=True,
        )

    def call(self, inputs):
        """Apply layer."""
        return tf.matmul(inputs, tf.expand_dims(self.w, 0))


if __name__ == "__main__":
    tf.enable_eager_execution()
    inputs = np.random.normal(size=(1, 10, 3))
    layer = MyLayer()
    outputs = layer(inputs)
    print(outputs.numpy())
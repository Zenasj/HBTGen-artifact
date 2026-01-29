# tf.random.uniform((B, 784), dtype=tf.float32) ← inferred input shape: MNIST images 28*28 flattened, batch size arbitrary

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self._layer1 = tf.keras.layers.Dense(20, activation='relu')
        self._layer2 = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        # Simple feedforward through two dense layers
        x = self._layer1(x)
        x = self._layer2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random batch of MNIST-like flattened images
    # Use batch size 32 as in original code
    batch_size = 32
    height = 28
    width = 28
    channels = 1  # MNIST grayscale
    
    # Flattened size: 28*28 = 784
    # Generate random uniform floats in [0,1), dtype float32
    x = tf.random.uniform((batch_size, height * width), dtype=tf.float32)
    return x


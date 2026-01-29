# tf.random.uniform((50000, 3072), dtype=tf.float32) ← CIFAR-10 flattened input shape

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CIFAR-10 images have 32x32x3 = 3072 features when flattened
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits for 10 classes

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights are initialized randomly by default
    return MyModel()

def GetInput():
    # Return a random tensor input with shape [50000, 3072] matching CIFAR-10 flattened data
    # Use float32 dtype as typical for TensorFlow models
    return tf.random.uniform((50000, 3072), dtype=tf.float32)


# tf.random.uniform((10, 10), dtype=tf.float32) ← input shape inferred from the DataGenerator __getitem__ output

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple dense network matching the example: input shape (10,), output shape (2)
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        out = self.dense2(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (batch_size=10, 10)
    # Matching the DataGenerator __getitem__ usage: batch size 10, feature 10, float32 dtype
    return tf.random.uniform((10, 10), dtype=tf.float32)


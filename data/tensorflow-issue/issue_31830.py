# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from Input(shape=(1,)) in the example

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Dense layer to mimic the example model
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    model = MyModel()
    # Compile with SGD optimizer and MSE loss as per the example
    model.compile(optimizer='SGD', loss='mean_squared_error')
    return model

def GetInput():
    # The example input shape is (batch_size, 1)
    # Using batch size 4 as a reasonable example for testing
    batch_size = 4
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


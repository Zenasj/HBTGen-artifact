# tf.random.uniform((1, 3), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer as per original example: input (3,) → output (3,)
        self.dense = tf.keras.layers.Dense(3, name='output')
    
    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel initialized with random weights
    return MyModel()

def GetInput():
    # Return a batch of size 1 with shape (1, 3) matching input shape of the model
    # Since model input shape is (None, 3), batch size 1 chosen arbitrarily to match example
    return tf.random.uniform((1, 3), dtype=tf.float32)


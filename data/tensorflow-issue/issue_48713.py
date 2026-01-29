# tf.random.uniform((B, 1), dtype=tf.float32) ← Input is batch of integer indices shaped (batch_size, 1)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A layer with weights of shape (10, 2)
        self.weight = self.add_weight(shape=(10, 2), initializer='random_normal', trainable=True)

    def call(self, inputs):
        # inputs shape: (batch_size, 1), expected to be integer indices in [0,9]
        indices = tf.cast(inputs, tf.int32)
        # Gather weights at the given indices along axis=0
        output = tf.gather(self.weight, axis=0, indices=tf.squeeze(indices, axis=-1))
        return output

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random batch of integer indices between 0 and 9, shape (batch_size, 1)
    # Assumed batch size 4 for demonstration; dtype float32 because model inputs convert to int64 internally
    batch_size = 4
    x = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=10, dtype=tf.int32)
    x = tf.cast(x, tf.float32)  # Input layer expects float type, cast back to int inside model
    return x


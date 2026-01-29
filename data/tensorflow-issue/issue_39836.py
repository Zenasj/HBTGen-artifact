# tf.random.uniform((B, ndim), dtype=tf.float32) ← Input shape is (batch_size, ndim) where ndim is feature dimension

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class MyModel(tf.keras.Model):
    def __init__(self, ndim):
        super().__init__()
        # Define the same sequential layers as in the reported issue
        self.model = Sequential([
            Dense(100, activation='relu', input_shape=(ndim,)),
            Dense(40, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=False):
        # Forward pass returns sigmoid predictions
        return self.model(inputs)


def my_model_function():
    # To instantiate MyModel, we need an input feature dimension
    # For demonstration, assume ndim=10 (since ndim is unknown from the issue)
    ndim = 10
    return MyModel(ndim)


def GetInput():
    # Generate a random tensor input matching expected shape (batch_size, ndim)
    # We choose batch_size=32 and ndim=10 to align with assumed model input dimension
    batch_size = 32
    ndim = 10
    return tf.random.uniform((batch_size, ndim), dtype=tf.float32)


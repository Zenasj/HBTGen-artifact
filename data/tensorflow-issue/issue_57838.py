# tf.random.uniform((1, 2, 2, 2), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D with filters=2, kernel_size=1, padding='same', dtype=tf.float32
        # as per the reported issue example
        self.conv = layers.Conv2D(
            filters=2, kernel_size=1, padding='same',
            dtype=tf.float32, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            # disabling autocast per example to avoid mixed precision issues
            # Note: tf.keras layers do not have autocast arg by default,
            # so this is simulated by controlling dtype directly.
        )

    @tf.function(jit_compile=True)
    def call(self, x):
        # Forward pass applies the conv2d layer.
        # The call method is jit-compiled to demonstrate XLA compatibility.
        y = self.conv(x)
        return y

def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()

def GetInput():
    # Return a random tensor matching the input used in the issue reproduction:
    # shape [1, 2, 2, 2], dtype float32
    # Using random uniform here to avoid constant inputs,
    # but it matches the requirements exactly.
    return tf.random.uniform((1, 2, 2, 2), dtype=tf.float32)


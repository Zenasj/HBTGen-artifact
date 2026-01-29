# tf.random.uniform((None,), dtype=tf.float32)  # Input shape inferred from original example: 1D tensor of variable length

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights or layers needed for the cos operation example

    @tf.function
    def call(self, x):
        # Directly compute cosine elementwise as the model does in the issue repro
        return tf.cos(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random 1D float32 tensor with variable batch size to simulate input signature [None]
    # Use batch size 8 as a reasonable default for test input
    batch_size = 8
    return tf.random.uniform((batch_size,), minval=-10.0, maxval=10.0, dtype=tf.float32)


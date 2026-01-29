# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape (batch_size, 1) for simplest Dense(1) model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model: InputLayer + Dense(1)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel, no pre-loaded weights needed for this minimal example
    return MyModel()

def GetInput():
    # Generate a random float32 tensor of shape (batch_size, 1)
    # Batch size is set to 8 as a reasonable default for testing
    batch_size = 8
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


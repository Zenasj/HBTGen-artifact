# tf.random.uniform((batch_size, 1), dtype=tf.float32) ← input shape inferred from generator yielding arrays of shape (1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a simple Dense layer with 3 output units as in the issue example
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs, training=False):
        # Forward pass through Dense layer
        return self.dense(inputs)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns a random input matching the generator output shape from the issue
    # The generator yields arrays of shape (1,), batched with batch size 32 in example
    # Since batching is 32, shape would be (32, 1)
    batch_size = 32
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


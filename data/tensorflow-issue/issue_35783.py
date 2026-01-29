# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from example: input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple dense layer matching the sample from the issue: Dense(1, input_shape=(1,))
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass through dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor compatible with input_shape=(1,)
    # Assuming batch size = 4 for example
    batch_size = 4
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


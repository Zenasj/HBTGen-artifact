# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape inferred from Dense layer input_shape=(3,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with input shape (3,) units=1 as per the original example
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Returns an instance of MyModel
    # No pretrained weights indicated, so default initialization
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input of shape (batch_size, 3)
    # Assume batch size 4 for example input
    return tf.random.uniform((4, 3), dtype=tf.float32)


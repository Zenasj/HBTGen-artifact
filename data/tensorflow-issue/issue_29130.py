# tf.random.uniform((B, 16), dtype=tf.float32) ← Input shape inferred from model input layer (shape=(16,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer with 1 output unit, linear activation by default
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass: input shape (B,16) -> output shape (B,1)
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape of MyModel's expected input:
    # The original example had input of shape (16,), so batch can be any size.
    # Here we pick batch size 8 as a reasonable default.
    return tf.random.uniform((8, 16), dtype=tf.float32)


# tf.random.uniform((B, 2), dtype=tf.float32) ← inferred input shape from Issue example: input_shape=[2] for Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Sequential-like structure from example:
        # Two Dense layers; first takes input of shape (2,), outputs 3 units,
        # second outputs 1 unit.
        self.dense1 = tf.keras.layers.Dense(3, input_shape=(2,))
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape of (batch_size, 2)
    # We choose batch size 4 arbitrarily for demonstration.
    batch_size = 4
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)


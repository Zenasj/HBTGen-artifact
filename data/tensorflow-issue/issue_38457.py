# tf.random.uniform((B, 2), dtype=tf.float16) ← inferred input shape is (batch_size, 2) as per XOR example inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A single Dense layer with 1 unit and tanh activation using float16 as in the issue reproducing code.
        # This layer corresponds to the model from the original issue.
        self.dense = tf.keras.layers.Dense(units=1, activation='tanh', dtype=tf.float16)

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel.
    # The weights will be randomly initialized by default.
    return MyModel()


def GetInput():
    # Return a float16 tensor shaped (1, 2) to replicate batch size of 1 and 2 features as per original XOR example.
    # Values between 0 and 1 (convertable binary inputs 0 or 1).
    # This works directly with MyModel.
    # Using tf.random.uniform to generate random input for versatility.
    return tf.random.uniform(shape=(1, 2), minval=0, maxval=1, dtype=tf.float16)


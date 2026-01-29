# tf.random.uniform((batch_size, lahead, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

# This model is an inferred reconstruction of the "stateless LSTM" model described in the original issue:
# - Input shape (batch_size=1, time_steps=lahead, feature_dim=1)
# - A single LSTM layer with 20 units followed by a Dense output with 1 unit
# The batch size and lahead (lookahead / timesteps) are assumed from the issue code:
batch_size = 1
lahead = 1  # inferred from code snippet; can be adjusted

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 20 units, stateless
        self.lstm = tf.keras.layers.LSTM(
            20,
            input_shape=(lahead, 1),
            batch_size=batch_size,
            stateful=False,  # matches the stateless model used in the example
            name='lstm_layer',
        )
        # Dense output layer producing one output value
        self.dense = tf.keras.layers.Dense(1, name='output_dense')

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, lahead, 1)
        x = self.lstm(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random uniform tensor input matching the required input shape (batch_size, lahead, 1)
    # Using dtype float32 as is typical for model inputs
    # Values approximately in the range [-0.1, 0.1] inferred from data generation in the issue
    return tf.random.uniform(
        shape=(batch_size, lahead, 1),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32
    )


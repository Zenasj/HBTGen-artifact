# tf.random.uniform((B, 2), dtype=tf.float32) ← Based on input shape (2,) in the example for fitting models

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture inferred from provided example:
        # Input layer shape=(2,), Dense(32), then Dense(1) output
        self.dense1 = tf.keras.layers.Dense(32, activation=None)
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Returns an instance of MyModel, weights are initialized randomly by default
    return MyModel()

def GetInput():
    # Returns a batch of inputs consistent with model input shape (batch_size, 2)
    # Using batch size = 32 as common default
    batch_size = 32
    # Random uniform float32 tensor, range [0,1)
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)


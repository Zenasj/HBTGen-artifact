# tf.random.uniform((B, 1), dtype=tf.float32)  # Input shape inferred from Input(shape=(1,))

import tensorflow as tf
from tensorflow import keras

# This model replicates the described minimal example:
# A simple Keras Model with a single Dense layer on input shape (1,)
# Demonstrates the model structure involved in the original issue:
# model = Model(input, Dense(250)(input))
# The issue context involved compiling & saving with Adam optimizer in TF v1 (graph) mode.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with output units 250, matching the original example
        self.dense_layer = keras.layers.Dense(250, name="dense_layer")
    
    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense_layer(inputs)

def my_model_function():
    # Build and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (?, 1), dtype float32 as expected by the model
    # Batch size chosen arbitrarily as 8 for demonstration
    batch_size = 8
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


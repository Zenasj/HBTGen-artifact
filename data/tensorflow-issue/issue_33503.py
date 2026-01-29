# tf.random.uniform((B, feature_dimension), dtype=tf.float32) ← Input shape is (batch_size, feature_dimension)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, feature_dimension=3):
        super().__init__()
        # Single Dense layer without activation (linear)
        self.dense = tf.keras.layers.Dense(1, activation=None)
        self.feature_dimension = feature_dimension

    def call(self, inputs, training=False):
        # Forward pass through Dense layer
        return self.dense(inputs)

def my_model_function():
    # Initialize model with default feature dimension 3 as in original example
    return MyModel(feature_dimension=3)

def GetInput():
    # Generate a random input tensor consistent with model input shape
    # Assumptions:
    # - batch_size is arbitrary, we choose 10 to match original example batch_size
    # - feature_dimension from model default is 3
    batch_size = 10
    feature_dimension = 3
    # Use tf.random.uniform for diverse input data
    return tf.random.uniform((batch_size, feature_dimension), dtype=tf.float32)


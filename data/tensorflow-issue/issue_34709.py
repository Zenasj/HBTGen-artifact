# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape is (batch_size, 10) with float features

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple 2-layer dense network
        self.dense1 = tf.keras.layers.Dense(80, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2)  # Output logits for 2 classes

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        logits = self.dense2(x)
        return logits


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a random input tensor with shape (batch_size=1, features=10)
    return tf.random.uniform((1, 10), dtype=tf.float32)


# Additional Notes / Assumptions:
# The original issue describes using this simple model within an Estimator with MirroredStrategy.
# The forwarded code snippet uses tf.keras.Sequential with 2 Dense layers having ReLU and then output logits.
# Labels are one-hot or two-class vectors of shape (batch_size, 2).
# Input tensor shape inferred from dataset is (batch_size, 10).
# For testing with XLA compilation, the model inputs should be tf.float32 tensors of shape (B, 10).

# This model is compatible with TF 2.20.0 and can be wrapped in a tf.function with jit_compile=True for XLA compilation.


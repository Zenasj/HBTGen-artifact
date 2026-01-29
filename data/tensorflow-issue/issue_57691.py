# tf.random.uniform((B, 1, 5, 12), dtype=tf.float32) ← Inferred input shape from the issue (batch_size, 1, 5, 12)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model applies a Dense layer on a rank-4 input (batch,1,5,12)
        # Dense layer expects last dimension as input_dim=12, output_dim=3
        # So the Dense layer will be applied to the last dimension of shape 12.
        # Keras Dense layer can broadcast over the previous dims (1,5).
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        # inputs shape: (batch_size, 1, 5, 12)
        # Apply dense layer to the last dimension.
        # The Dense layer from Keras applies on the last axis,
        # preserving the other axes: output shape (batch,1,5,3).
        return self.dense(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Create a random float tensor with the appropriate shape and dtype
    # According to the issue, batch size is dynamic, let's set batch_size=2 for example
    batch_size = 2
    shape = (batch_size, 1, 5, 12)
    # Use uniform float in [0,1) to simulate normalized input (like noise/255)
    # dtype float32 to match Keras default
    return tf.random.uniform(shape, dtype=tf.float32)


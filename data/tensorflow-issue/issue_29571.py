# tf.random.uniform((1, 64), dtype=tf.float32) ← Input is a 1D tensor with shape (1, 64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No extra layers needed; model just does an element-wise square, then slices every second element.
        # Slicing is done along the last dimension (channels).
    
    def call(self, inputs):
        x = inputs
        x = x * x
        # Slice to keep every second element along the last axis, equivalent to x[:, ::2]
        x = x[:, ::2]
        return x

def my_model_function():
    # Return an instance of MyModel with no additional initialization or weights.
    return MyModel()

def GetInput():
    # Returns a random float32 tensor of shape (1, 64) matching the model input
    return tf.random.uniform((1, 64), dtype=tf.float32)


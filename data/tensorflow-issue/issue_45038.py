# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model is a single Multiply layer taking input * input
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        # inputs shape: (B, 1)
        # multiply input by itself element-wise, output same shape
        return self.multiply([inputs, inputs])

def my_model_function():
    # Return an instance of the model with single multiply layer
    return MyModel()

def GetInput():
    # Based on the issue, the input shape is (B, 1)
    # Provide a representative input in the problematic range [1.3, 1.4]
    # Using batch size 2 with two inputs to represent the representative dataset calls
    import numpy as np
    # Create a batch of shape (2,1), corresponding to samples 1.3 and 1.4
    example_inputs = tf.constant([[1.3], [1.4]], dtype=tf.float32)
    return example_inputs


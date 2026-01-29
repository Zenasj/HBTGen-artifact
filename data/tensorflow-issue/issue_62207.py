# tf.random.uniform((10, 9, 8, 6), dtype=tf.bfloat16) ← inferred input shape and dtype from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No extra layers; just raw ops usage as per the issue.

    @tf.function(jit_compile=True)
    def call(self, x):
        # Perform Cos then Asinh using raw_ops, exactly as in the issue.
        y = tf.raw_ops.Cos(x=x)
        y = tf.raw_ops.Asinh(x=y)
        return y

def my_model_function():
    # Return an instance of MyModel for usage/testing.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input shape and dtype used in the example.
    # Shape: [10, 9, 8, 6], dtype: tf.bfloat16, range [-100, 100]
    return tf.random.uniform([10, 9, 8, 6], minval=-100, maxval=100, dtype=tf.bfloat16)


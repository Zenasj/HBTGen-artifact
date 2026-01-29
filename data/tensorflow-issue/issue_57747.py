# tf.constant([-0.1982182], shape=(1,), dtype=tf.float32) ← input shape and dtype inferred from the issue's example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed; just use raw ops per the issue

    @tf.function(jit_compile=True)
    def call(self, x):
        # Replicating the logic from the issue:
        # - Compute y = x / x (division)
        # - Compute z = x ** y (power)
        # The issue is that on XLA-GPU, y can be imprecise leading to nan in z.
        # We output both y and z so that a caller or test can inspect intermediate imprecision.
        y = tf.divide(x, x)
        z = tf.pow(x, y)
        return y, z

def my_model_function():
    # Just return an instance of MyModel (no weight init needed)
    return MyModel()

def GetInput():
    # Return the tensor used in all examples from the issue: a 1D tensor with one float value.
    # Using dtype float32 as in the original.
    return tf.constant([-0.1982182], dtype=tf.float32)


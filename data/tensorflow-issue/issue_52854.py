# tf.random.uniform((B, 5), dtype=tf.float32) ← inferred input shape: batch size unknown, feature size 5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters, just performs element-wise abs
        # We'll implement the abs operation to match the Keras model behavior.
        # Since the original issue talks about quantized ABS misbehavior,
        # here we just implement the regular abs as a demonstration.
    
    def call(self, inputs):
        # inputs shape: (batch_size, 5)
        # return absolute value element-wise
        return tf.math.abs(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a batch of random negative values only, shape (1,5) to emulate the representative dataset in the issue
    # Using uniform distribution from -1.0 to -0.01 to mimic negative values only input as per issue
    batch_size = 1
    feature_size = 5
    return tf.random.uniform(shape=(batch_size, feature_size), minval=-1.0, maxval=-0.01, dtype=tf.float32)


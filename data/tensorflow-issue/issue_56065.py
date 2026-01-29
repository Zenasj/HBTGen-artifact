# tf.random.uniform((1, 512, 512, 3), dtype=tf.float16) ← input shape and dtype inferred from issue example

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable layers here — just a wrapper around tf.image.total_variation
        # Since total_variation returns a scalar per image, output shape is (batch_size,)
        
    def call(self, inputs):
        # inputs is a float16 tensor of shape (1, 512, 512, 3)
        # tf.image.total_variation returns a float32 tensor, so we must decide on output dtype
        # From the issue, outputting float16 causes NaN in total_variation output,
        # so we cast result back to float32 here inside the call.
        
        # Compute total variation on inputs, always in float32 to avoid NaN issues
        # Cast input to float32 before total_variation
        inputs_fp32 = tf.cast(inputs, tf.float32)
        tv = tf.image.total_variation(inputs_fp32)  # shape: (batch_size,)
        # Return float32 output for stability
        return tv

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float16 tensor of shape (1, 512, 512, 3) with values in a range
    # Using uniform distribution to simulate image pixel values scaled between 0 and 1
    return tf.random.uniform(shape=(1, 512, 512, 3), minval=0.0, maxval=1.0, dtype=tf.float16)


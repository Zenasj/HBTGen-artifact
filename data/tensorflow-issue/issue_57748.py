# tf.random.uniform((1, 1, 3, 2), dtype=tf.float64) ← Input shape inferred from the issue's example code

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layer with 2 output filters, kernel size 1x1, no padding, float64 dtype, autocast disabled
        self.conv = layers.Conv2D(
            filters=2,
            kernel_size=1,
            padding='valid',
            dtype=tf.float64,
            autocast=False
        )
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Mimic the original pipeline:
        # 1. floor input
        # 2. apply conv on floored input
        # 3. add conv output and floored input
        x = tf.floor(inputs)
        conv_out = self.conv(x)
        out = tf.add(conv_out, x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching (1, 1, 3, 2) shape with dtype float64
    # This matches the example input in the issue:
    # tf.constant(3.14, shape=[1,1,3,2], dtype=tf.float64)
    # Using uniform random input for generality
    return tf.random.uniform(shape=(1, 1, 3, 2), dtype=tf.float64, minval=0.0, maxval=10.0)


# tf.random.uniform((B, 128, 64, 2), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Conv2D layer matching original example: kernel size=1, output channels=2, no activation
        self.conv = tf.keras.layers.Conv2D(filters=2, kernel_size=1, activation=None, 
                                           data_format='channels_last')
    
    def call(self, inputs, training=False):
        # Forward pass: apply conv2d layer
        return self.conv(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # The model expects input shape (batch_size, 128, 64, 2), dtype float32
    batch_size = 16  # same as in example
    height = 128
    width = 64
    channels = 2
    # Generate random input tensor in range [0,1), float32
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)


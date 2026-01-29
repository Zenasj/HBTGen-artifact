# tf.random.uniform((B, 2), dtype=tf.int32) ← Input is a batch of 2-element int32 vectors

import tensorflow as tf

class TwoOutputsLayer(tf.keras.layers.Layer):
    def call(self, x):
        # returns two tensors, each same shape as input
        return x + 1, x - 1

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.two_outputs_layer = TwoOutputsLayer()
    
    def call(self, inputs):
        # Return tuple of outputs matching original model's multiple outputs
        out1, out2 = self.two_outputs_layer(inputs)
        return out1, out2

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random int32 tensor of shape (batch_size, 2), matching input shape used in example
    # Use batch size 4 by default
    batch_size = 4
    return tf.random.uniform((batch_size, 2), minval=0, maxval=10, dtype=tf.int32)


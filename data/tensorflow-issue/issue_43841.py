# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape is (batch_size, 1), as input is (1,)

import tensorflow as tf
from tensorflow.python import keras as pykeras
from tensorflow.python.framework import ops
import copy
import gc

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define input layer shape (1,)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(1,))
        # Dense layer with 1 unit (matching example)
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        """
        Forward pass replicating the example logic:
        1. Apply dense layer to input
        2. Slice output with [:, :-1], which for 1 unit results in empty shape (since last axis size is 1)
           So output shape is (batch_size, 0) (empty slice)
        
        This matches the snippet:
          model_output = dense(input)[:,:-1]
        
        Note: This is unusual, but reflects the original code.
        """
        x = self.input_layer(inputs)
        x = self.dense(x)
        # Slice to exclude last feature dimension
        x = x[:, :-1]
        return x

def my_model_function():
    """
    Returns an instance of MyModel.
    This model reflects construction and slicing as in the issue example.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor matching the shape expected by MyModel:
    (batch_size, 1) float32 tensor.
    Batch size is chosen as 4 for example.
    """
    batch_size = 4
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


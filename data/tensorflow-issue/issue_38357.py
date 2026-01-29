# tf.random.uniform((BATCH, 784), dtype=tf.float32) ← From example input shape for the MNIST model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# This model is a reproduction of the key relevant example from the GitHub issue:
# a simple MLP with two Dense layers: 4096 units with relu + 1 output unit with relu
# Input: flattened MNIST images with shape (batch_size, 784)
# This model was used to demonstrate loss overflow in mixed precision training and
# the need to allow dynamic loss scale smaller than one.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(4096, activation='relu', name='dense_1')
        self.dense2 = layers.Dense(1, activation='relu', name='dense_2')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel.
    # Note: The original issue used mixed precision with loss scaling to handle float16 training.
    # The model can be used with or without mixed precision, but here's the standard model.
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected shape: (batch_size, 784)
    # Using a batch size of 32 for general use.
    batch_size = 32
    # Inputs in example are float32 normalized flattened MNIST images.
    # Typical range is [0, 1], so uniform over that range.
    return tf.random.uniform((batch_size, 784), minval=0., maxval=1., dtype=tf.float32)


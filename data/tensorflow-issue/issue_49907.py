# tf.random.uniform((B, 13), dtype=tf.float32) ← The input has 13 features based on the Boston housing dataset

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two hidden layers with 10 units and SELU activation, LeCun normal initializer
        self.dense1 = tf.keras.layers.Dense(
            10, activation='selu',
            kernel_initializer=tf.keras.initializers.lecun_normal()
        )
        self.dense2 = tf.keras.layers.Dense(
            10, activation='selu',
            kernel_initializer=tf.keras.initializers.lecun_normal()
        )
        # Output layer: single linear unit for regression
        self.out = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

def my_model_function():
    # Returns a compiled MyModel for regression with MSE loss and Adam optimizer.
    model = MyModel()
    # Compile with Adam optimizer and MSE loss as per the original code
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model

def GetInput():
    # Return a random input tensor matching the shape: batch size 4, 13 features
    # The original batch size used was 4, and feature size is 13 from Boston data
    return tf.random.uniform(shape=(4, 13), dtype=tf.float32)


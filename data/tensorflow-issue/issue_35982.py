# tf.random.uniform((32, 10), dtype=tf.float32) ← Input shape inferred from Input layer in code

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model layers per the original issue code:
        # Input shape is (10,)
        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.dense2 = tf.keras.layers.Dense(20, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(1, name='output')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        out = self.output_layer(x)
        return out

def my_model_function():
    # Returns an instance of MyModel with no additional weight loading
    return MyModel()

def GetInput():
    # Returns a random tensor with shape (32, 10) matching the input expected by MyModel
    # Using tf.random.uniform with default float32 dtype
    return tf.random.uniform((32, 10), dtype=tf.float32)


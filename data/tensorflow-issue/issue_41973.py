# tf.random.uniform((2000, 42, 3), dtype=tf.float32) ← inferred input shape from one_hot encoding of states with rows=6, columns=7, depth=3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(6*7, 3))  # input_shape = rows*columns, depth=3
        self.dense = tf.keras.layers.Dense(512)

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 42, 3)
        x = self.flatten(inputs)  # (batch_size, 42*3=126)
        x = self.dense(x)         # (batch_size, 512)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate random input tensor compatible with MyModel
    # Shape is (batch_size, rows*columns, depth), consistent with one_hot encoded states in the issue
    batch_size = 2000  # as used in the issue example
    rows = 6
    columns = 7
    depth = 3
    inputs = tf.random.uniform((batch_size, rows * columns, depth), dtype=tf.float32)
    return inputs


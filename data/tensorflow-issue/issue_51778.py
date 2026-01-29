# tf.random.uniform((10, 10), dtype=tf.float32) for inputs, matching the example dataset tensors

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple single Dense layer model as defined by the user example
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the example dataset: shape (10, 10), float32
    # This mimics the input used in dataset_fn from the issue
    return tf.random.uniform((10, 10), dtype=tf.float32)


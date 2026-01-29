# tf.random.uniform((None, 32), dtype=tf.float32) ← inferred input shape is (batch_size, 32) from the Input layer shape in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple single Dense layer model matching the example in the issue
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    # Weight initialization is default (Glorot uniform)
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size, 32)
    # Use batch size 8 as a reasonable default for demonstration
    return tf.random.uniform((8, 32), dtype=tf.float32)


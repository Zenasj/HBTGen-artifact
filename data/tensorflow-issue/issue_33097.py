# tf.random.uniform((32, 4), dtype=tf.float32) ← inferred input shape from example X shape (32,4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple model matching the example: input shape (4,), output scalar with sigmoid activation
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel (untrained weights)
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (32, 4) matching the example batch input
    # Using float32 as typical for TF models
    return tf.random.uniform((32, 4), dtype=tf.float32)


# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape from keras.layers.Dense(1, input_shape=(1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original example is a simple Sequential model with one Dense layer of 1 output unit:
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))

    def call(self, inputs, training=False):
        # Forward pass through the Dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size, features) matching input_shape=(1,)
    # Assume batch size 4 for example
    return tf.random.uniform((4, 1), dtype=tf.float32)


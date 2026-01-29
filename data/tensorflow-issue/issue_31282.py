# tf.random.uniform((B, 5, 10), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Our example model from the issue: input shape (None, 5, 10), Dense(10) layer applied to each time step
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        # inputs shape: (batch, 5, 10)
        # Dense layer expects last dimension 10, applied independently along sequence dimension
        # Keras Dense with 3D input applies the same dense layer on last axis
        return self.dense(inputs)

def my_model_function():
    # Returns an instance of MyModel; weights initialized randomly
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with MyModel
    # Shape: (batch_size, 5, 10)
    # Using batch size 4 as an example arbitrary choice
    return tf.random.uniform((4, 5, 10), dtype=tf.float32)


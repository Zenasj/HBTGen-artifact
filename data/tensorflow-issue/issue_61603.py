# tf.random.uniform((3, 74, 74, 256), dtype=tf.float32) ← input shape from issue reproduction code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a tf.compat.v1.keras.layers.MaxPool2D layer using the parameters from the issue
        # Note: pool_size has extremely large float and large int values, which likely triggers the error.
        # We keep the original parameters to reproduce the scenario.
        self.maxpool = tf.compat.v1.keras.layers.MaxPool2D(
            pool_size=[1e+38, 1048576],
            strides=[2, 2],
            padding="same",
            data_format=None,
        )

    def call(self, inputs):
        # inputs is expected to be a 4-D tensor (batch, height, width, channels)
        # Pass through the MaxPool2D layer
        return self.maxpool(inputs)

def my_model_function():
    # Return an instance of MyModel with the MaxPool2D layer initialized
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input for MyModel
    # Shape: [3, 74, 74, 256], dtype: float32 (from provided reproducing code)
    return tf.random.uniform([3, 74, 74, 256], dtype=tf.float32)


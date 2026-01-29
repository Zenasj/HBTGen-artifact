# tf.random.uniform((3, 4), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class ScalarLayer(tf.keras.layers.Layer):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        # Using a trainable scalar variable wrapped in a Layer so it appears in model.layers and saves properly in H5
        self.value = tf.Variable(value, dtype=tf.float32, trainable=True)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use ScalarLayer encapsulating a scalar trainable variable, which works well with h5 saving
        self.hack = ScalarLayer(1.0)
        self.something = self.hack.value

    def call(self, x):
        # Identity call to match expected usage, input shape assumed (3,4) per examples
        return x


def my_model_function():
    # Return an instance of MyModel with initialized scalar variable
    return MyModel()


def GetInput():
    # Return a random tensor with shape (3, 4), dtype float32 to match the example inputs used in the issue
    return tf.random.uniform((3, 4), dtype=tf.float32)


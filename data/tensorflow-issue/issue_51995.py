# tf.random.uniform((1, 300, 300, 3), dtype=tf.float32) ← inferred input shape from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Conv2D layer similar to the user's dummy_net example with 6 filters, kernel size 3
        self.conv = tf.keras.layers.Conv2D(6, kernel_size=3)

    def call(self, x):
        x = self.conv(x)
        return x

def my_model_function():
    # Return an instance of MyModel with the default Conv2D weights initialization
    return MyModel()

def GetInput():
    # Generate a random tensor matching the input shape used in the original issue (batch=1, 300x300, 3 channels)
    # The dtype is float32 as typical for Conv2D inputs
    return tf.random.uniform((1, 300, 300, 3), dtype=tf.float32)


# tf.random.uniform((1, 300, 300, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Conv2D layer as per issue's dummy_net.py example
        self.conv = tf.keras.layers.Conv2D(filters=6, kernel_size=3)

    def call(self, x):
        # Simple forward pass through single Conv2D layer
        return self.conv(x)

def my_model_function():
    # Instantiate and return the model instance
    return MyModel()

def GetInput():
    # Return a random input tensor matching model's expected input shape
    # Shape: batch size = 1, height = 300, width = 300, channels = 3 (e.g. RGB image)
    return tf.random.uniform(shape=(1, 300, 300, 3), dtype=tf.float32)


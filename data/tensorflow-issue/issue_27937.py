# tf.random.uniform((1, 32, 32, 3), dtype=tf.float32) ← Inferred input shape from example usage in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model described: Sequential with two Conv2D layers
        self.conv1 = tf.keras.layers.Conv2D(256, 3, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(3, 3, padding="same")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(x)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped (1, 32, 32, 3) as input to model,
    # matching the example usage and common image format (batch, height, width, channels).
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)


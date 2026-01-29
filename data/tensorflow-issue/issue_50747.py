# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  # Input shape based on example: batch size unspecified (B), height=32, width=32, channels=3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D with kernel size 3x3, dilation rate 2, filters=3, no bias (to separate bias handling explicitly)
        # However, original example does not specify bias=False; default is bias=True.
        # We keep bias=True to replicate the original model logic that showed the issue.
        self.conv = tf.keras.layers.Conv2D(
            filters=3, kernel_size=3, dilation_rate=2, padding='valid', use_bias=True
        )

    def call(self, inputs):
        # Forward pass through Conv2D layer
        x = self.conv(inputs)
        return x


def my_model_function():
    # Return an instance of MyModel initialized with default weights
    return MyModel()


def GetInput():
    # Return random input tensor matching the model input: (batch_size=1, height=32, width=32, channels=3)
    # Use uniform distribution, dtype float32, to match example input type
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)


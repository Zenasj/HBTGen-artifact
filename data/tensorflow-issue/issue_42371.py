# tf.random.uniform((1, 100, 1, 512), dtype=tf.float32) ← Input shape inferred from example usage in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, channels=256, ksize=16, stride=8, padding="same"):
        super(MyModel, self).__init__()
        # This class emulates a Conv1DTranspose using Conv2DTranspose by expanding dims
        # In the original code, the input was 4D: (batch, length, 1, channels_in)
        # This layer applies Conv2DTranspose with kernel size (ksize, 1) and stride (stride, 1)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=channels,
            kernel_size=(ksize, 1),
            strides=(stride, 1),
            padding=padding,
        )

    def call(self, x):
        # Expecting input shape: (batch, length, 1, channels_in)
        # The original code did not use explicit expand_dims or squeeze here,
        # so input must already be 4D with 2nd dim = length, 3rd dim = 1
        x = self.conv1d_transpose(x)
        return x


def my_model_function():
    # Return an instance of MyModel with default parameters used in the issue example
    return MyModel(channels=256, ksize=16, stride=8, padding="same")


def GetInput():
    # Return a random tensor matching the input expected by MyModel
    # From the issue example:
    # shape = (1, 100, 1, 512), dtype = float32
    return tf.random.uniform((1, 100, 1, 512), dtype=tf.float32)


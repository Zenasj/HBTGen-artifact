# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← assuming input images of shape (32, 128, 3) as in resize step

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a data augmentation pipeline equivalent to what was suggested in the issue:
        # RandomFlip, RandomRotation, and RandomZoom as an example, configurable as needed.
        self.data_augmentation = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=(-0.052, 0.035)),  # roughly ±5 degrees in radians (≈ ±0.087 rad)
            layers.RandomZoom(height_factor=0.1, width_factor=0.1)
        ])

    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, 32, 128, 3), float32 in [0, 1]
        if training:
            # Apply data augmentation only during training
            return self.data_augmentation(inputs, training=True)
        else:
            # During inference, pass inputs as-is
            return inputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input:
    # Batch size randomly set to 8 for example,
    # Image size 32x128, 3 channels, dtype float32 in [0, 1]
    batch_size = 8
    height = 32
    width = 128
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), minval=0.0, maxval=1.0, dtype=tf.float32)


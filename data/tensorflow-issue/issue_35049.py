# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use ResNet50V2 from keras applications as the main model backbone
        # Input shape matches CIFAR-10 images: (32,32,3)
        self.backbone = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights=None,
            input_shape=(32, 32, 3),
            pooling='max',
            classes=10)

    def call(self, x, training=False):
        # Forward pass through backbone model
        return self.backbone(x, training=training)


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor simulating a batch of images with shape (batch_size, 32, 32, 3)
    # Choose a default batch size similar to issue example (100) for representative input
    batch_size = 100
    # Generate random float32 tensor with values between 0 and 255 (typical image pixel range)
    return tf.random.uniform((batch_size, 32, 32, 3), minval=0, maxval=255, dtype=tf.float32)


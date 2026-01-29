# tf.random.uniform((1, 3, 5000, 5000), dtype=tf.float32)
import tensorflow as tf
import numpy as np
import time


class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(3, 5000, 5000), num_classes=15):
        """
        This model replicates a ResNet50 constructed with channels_first inputs,
        num_classes outputs, and no pretrained weights.

        The model handles inputs of shape (batch, channels=3, height=5000, width=5000).

        Note:
          - This large input size is intended to reproduce the OOM behavior seen in the issue.
          - The original issue used tf.keras.applications.ResNet50 with channels_first format.
        """
        super().__init__()

        # Enforce channels_first data format globally in Keras backend
        tf.keras.backend.set_image_data_format('channels_first')
        self.input_shape_for_model = input_shape
        self.num_classes = num_classes

        # Build a ResNet50 instance with no weights, channels_first
        # The keras ResNet50 implementation expects channels_last by default,
        # so we must specify input_shape with channels_first.
        self.base_model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=self.input_shape_for_model,
            classes=self.num_classes
        )

    def call(self, inputs, training=False):
        """
        Forward pass: pass input to the ResNet50 model and return logits.
        """
        return self.base_model(inputs, training=training)


def my_model_function():
    """
    Create and return an instance of MyModel initialized for a 3x5000x5000 input with 15 classes.
    """
    return MyModel(input_shape=(3, 5000, 5000), num_classes=15)


def GetInput():
    """
    Generate a random input batch that matches the expected input shape:
    batch size = 1, channels=3, height=5000, width=5000.
    Values are float32 random normal.

    This mimics the large input tensor that triggers OOM in the issue.
    """
    batch_size = 1
    input_shape = (3, 5000, 5000)
    # Use tf.random.uniform since the shape is known and float32 expected
    input_tensor = tf.random.uniform(
        shape=(batch_size,) + input_shape,
        minval=0,
        maxval=1,
        dtype=tf.float32
    )
    return input_tensor


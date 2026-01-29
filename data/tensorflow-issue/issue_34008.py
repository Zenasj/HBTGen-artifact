# tf.random.uniform((B, 10, 200, 200, 128), dtype=tf.float32) ← inferred input shape from the issue

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, Conv2D


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv3D layers as per original code inside strategy.scope()
        self.conv3d_1 = Conv3D(64, kernel_size=3, strides=1, padding='same')
        self.conv3d_2 = Conv3D(64, kernel_size=3, strides=1, padding='same')
        self.conv3d_3 = Conv3D(64, kernel_size=3, strides=1, padding='same')
        # Conv2D layer after max pooling
        self.conv2d = Conv2D(14, kernel_size=3, padding='same')
    
    def call(self, inputs, training=False):
        """
        inputs: tensor of shape (batch_size, 10, 200, 200, 128)
        returns: tensor of shape (batch_size, 200, 200, 14)
        """
        x = self.conv3d_1(inputs)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        # Reduce max over axis=1 (the temporal dimension)
        x = tf.reduce_max(x, axis=1)
        out = self.conv2d(x)
        return out


def my_model_function():
    """
    Creates and returns an instance of MyModel.
    This mimics the creation of the model inside the MirroredStrategy.scope().
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor compatible with MyModel's expected input:
    batch shape = (global_batch_size, 10, 200, 200, 128),
    where global_batch_size is inferred for demo as 4 (like BATCH_SIZE_PER_SYNC).
    dtype = tf.float32
    """
    BATCH_SIZE_PER_SYNC = 4

    # Here we assume a single replica: batch size = BATCH_SIZE_PER_SYNC * 1
    # In distributed training, batch size would be multiplied by num replicas.
    # For simplicity, use batch size of 4 here.
    batch_size = BATCH_SIZE_PER_SYNC

    # Shape: (batch_size, 10, 200, 200, 128)
    return tf.random.uniform(
        shape=(batch_size, 10, 200, 200, 128), dtype=tf.float32
    )


# tf.random.uniform((B, 10, 200, 200, 128), dtype=tf.float32) ← inferred input shape from original issue fake data

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv2D
from tensorflow.keras import Model, Input

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct Conv3D stack as in original example 
        self.conv3d_1 = Conv3D(64, kernel_size=3, strides=1, padding='same')
        self.conv3d_2 = Conv3D(64, kernel_size=3, strides=1, padding='same')
        self.conv3d_3 = Conv3D(64, kernel_size=3, strides=1, padding='same')
        # Conv2D layer after reduce_max
        self.conv2d = Conv2D(14, kernel_size=3, padding='same')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        # inputs shape: (B, 10, 200, 200, 128)
        x = self.conv3d_1(inputs)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        # max pool over temporal dimension axis=1 (the 10 frames)
        x = tf.reduce_max(x, axis=1)  # shape: (B, 200, 200, 64)
        out = self.conv2d(x)  # shape: (B, 200, 200, 14)
        return out


def my_model_function():
    # Instantiate and return the model
    model = MyModel()
    # Build the model by calling with dummy input to create weights (optional)
    dummy_input = tf.zeros((1, 10, 200, 200, 128), dtype=tf.float32)
    _ = model(dummy_input)
    return model


def GetInput():
    # Return a random tensor matching the expected input shape
    # We use tf.random.uniform with dtype float32 as input in the issue is float32
    # Batch size B=4 chosen based on BATCH_SIZE_PER_SYNC in original code,
    # could be any positive integer.
    B = 4
    input_tensor = tf.random.uniform((B, 10, 200, 200, 128), dtype=tf.float32)
    return input_tensor


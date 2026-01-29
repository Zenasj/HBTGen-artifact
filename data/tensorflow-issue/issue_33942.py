# tf.random.uniform((32, 64, 64, 3), dtype=tf.float64)
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example from the issue:
        # Two Conv2D layers with 6 filters, (8,8) kernel, stride 2, relu activation
        # Flatten then Dense with 6 output units and softmax activation
        # Use float64 dtype to match the issue setup
        self.conv1 = Conv2D(
            filters=6, kernel_size=(8, 8), strides=(2, 2), activation='relu', dtype=tf.float64)
        self.conv2 = Conv2D(
            filters=6, kernel_size=(8, 8), strides=(2, 2), activation='relu', dtype=tf.float64)
        self.flatten = Flatten(dtype=tf.float64)
        self.dense = Dense(6, activation='softmax', dtype=tf.float64)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        out = self.dense(x)
        return out


def my_model_function():
    # Return an instance of MyModel, weights will be randomly initialized.
    # Note: The original reproducibility issue mentions seeds but that is outside this scope.
    # This instance is ready for use and can be compiled/train as needed.
    return MyModel()


def GetInput():
    # Return a random tensor input matching the input expected by MyModel:
    # Batch size 32, height 64, width 64, channels 3
    # Use float64 since model layers expect it for reproducibility issues in the original code
    return tf.random.uniform((32, 64, 64, 3), dtype=tf.float64)


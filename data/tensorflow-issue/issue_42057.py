# tf.random.uniform((1, 10, 5, 1), dtype=tf.float32) ← inferred input shape from the issue Keras model examples

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two sub-models as seen in the issue:
        # 1) Model with Conv2D + Flatten + Dense (which triggers threading in TFLite)
        # 2) Model without Conv2D (only Flatten + Dense)

        # Model with convolution path
        self.conv_model = tf.keras.Sequential([
            InputLayer(input_shape=(10, 5, 1)),
            Conv2D(32, (3, 3)),
            Flatten(),
            Dense(1)
        ])

        # Model without convolution path
        self.no_conv_model = tf.keras.Sequential([
            InputLayer(input_shape=(10, 5, 1)),
            Flatten(),
            Dense(1)
        ])

    def call(self, inputs):
        # Forward through both sub-models
        out_conv = self.conv_model(inputs)
        out_no_conv = self.no_conv_model(inputs)

        # Compare outputs with a tolerance and return a boolean tensor indicating close outputs
        # This illustrates a fusion of the two models with comparison logic,
        # which lines up with the requirement to integrate multiple related models and implement comparison.

        # Use tf.reduce_all over last dimension after tf.abs diff <= tolerance
        tolerance = 1e-5
        is_close = tf.reduce_all(
            tf.abs(out_conv - out_no_conv) <= tolerance,
            axis=-1,
            keepdims=True
        )
        return is_close


def my_model_function():
    # Return a fully initialized instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input shape expected by the model
    # Based on example inputs: batch dimension added as 1 by default
    return tf.random.uniform((1, 10, 5, 1), dtype=tf.float32)


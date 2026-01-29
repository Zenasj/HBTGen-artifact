# tf.random.uniform((10, 100, 100, 4), dtype=tf.float32) ← inferred input shape from example in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A model demonstrating convolutional and dense layers as described in the issue comments/test scripts.

    This model fuses two example usages given in the issue discussion:
    - A Conv2D layer applied to a 4D input tensor of shape (B, H, W, C)
    - A Sequential Dense model on 2D input.

    The forward method applies both submodels and returns their outputs as a tuple.

    This design helps illustrate the profiling scenario discussed in the issue:
    - Conv2D example: inputs shape (B,H,W,C) = (10,100,100,4)
    - Dense example: inputs shape (B,features) = (100,4)
    """

    def __init__(self):
        super().__init__()
        # Conv2D model (from Chunk 2 example)
        self.conv = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation=None,
        )
        # Dense sequential model (from Chunk 3 example)
        self.dense_model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation=None),
            tf.keras.layers.Dense(1, activation=None),
        ])

    def call(self, inputs, training=False):
        # Expecting inputs as a tuple: (conv_input, dense_input)
        conv_input, dense_input = inputs

        # Apply conv model
        conv_output = self.conv(conv_input)

        # Apply dense model
        dense_output = self.dense_model(dense_input)

        # Return both outputs as a tuple
        return conv_output, dense_output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate inputs matching the expected inputs for MyModel.call
    # conv_input shape: (10, 100, 100, 4) - batch size 10, height 100, width 100, channels 4
    conv_input = tf.random.uniform((10, 100, 100, 4), dtype=tf.float32)

    # dense_input shape: (100, 4) - batch size 100, features 4
    dense_input = tf.random.uniform((100, 4), dtype=tf.float32)

    # Return tuple matching model call signature
    return conv_input, dense_input


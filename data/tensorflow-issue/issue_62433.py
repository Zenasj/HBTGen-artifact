# tf.random.uniform((2, 15, 4, 4), dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using DepthwiseConv2D with kernel_size=3x3, strides=2x2, padding='valid'
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            depth_multiplier=1,
            activation=None,
            use_bias=False,
            depthwise_initializer='glorot_uniform',
            bias_initializer='zeros',
            dtype=tf.float64  # matching input dtype
        )

    def call(self, inputs, training=False):
        # Forward pass through the DepthwiseConv2D layer
        return self.depthwise_conv(inputs)

def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input of shape [batch, height, width, channels]
    # According to the issue analysis:
    # - Input shape: batch=2, height=15, width=4, channels=4 (channels inferred from original [2,15,4,4])
    # - dtype: tf.float64 as per original code
    # Values are sampled uniformly between 0 and 1 (not zero) to allow gradients to flow properly.
    return tf.random.uniform([2, 15, 4, 4], minval=0, maxval=1, dtype=tf.float64)


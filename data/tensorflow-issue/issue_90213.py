# tf.random.uniform((B, L, C), dtype=tf.float32) ← inferred input shape is (batch, length, channels) for SeparableConv1D

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The official tf.keras.layers.SeparableConv1D does not support 'causal' padding.
        # To emulate causal padding behavior, we implement manual left padding before a valid padding conv.
        # Assumptions:
        # - kernel_size=2 (from example)
        # - dilation_rate=1
        # - filters=3
        # - input channels=3 (from example)
        # Input shape expected: (batch, length, channels)
        self.kernel_size = 2
        self.dilation_rate = 1
        self.filters = 3

        # Depthwise + Pointwise convs mimic SeparableConv1D
        # We will use padding='valid' and do manual causal padding ourselves.

        # Depthwise convolution with depth_multiplier=1
        self.depthwise_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=self.kernel_size,
            strides=1,
            padding='valid',
            dilation_rate=self.dilation_rate,
            depth_multiplier=1,
            activation=None,
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        # Pointwise convolution to map depthwise channels to filters
        self.pointwise_conv = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            padding='valid',
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform'
        )

    def call(self, inputs):
        # inputs shape: (batch, length, channels)
        # Implement causal padding manually
        # causal padding for kernel_size=2 means 1 zero-pad on left for dilation=1
        pad_len = (self.kernel_size - 1) * self.dilation_rate
        # Pad left side along time dimension (axis=1)
        paddings = tf.constant([[0,0], [pad_len,0], [0,0]])
        x_padded = tf.pad(inputs, paddings, mode='CONSTANT', constant_values=0)

        # Apply depthwise conv + pointwise conv
        x = self.depthwise_conv(x_padded)
        x = self.pointwise_conv(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Following example input: batch size 4, length 10, channels 3 (channels last for Conv1D)
    return tf.random.uniform((4, 10, 3), dtype=tf.float32)


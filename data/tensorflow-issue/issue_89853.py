# tf.random.uniform((1, 4, 4, 1), dtype=tf.float32) ← inferred input shape from input_sizes: [1,4,4,1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # DepthwiseConv2D filter hardcoded as in the example (1x1 filter)
        # Weights shape for depthwise conv2d filter expected by TF is [filter_height, filter_width, in_channels, channel_multiplier]
        # The example used a scalar [[1.0]], so we assume 1 input channel, channel_multiplier=1
        self.filter = tf.constant([[[[1.0]]]], dtype=tf.float32)
        # Strides and padding as per example
        self.strides = [1, 1]
        self.padding = 'VALID'
        # The input_sizes shape is fixed here based on the example
        self.input_sizes = tf.constant([1, 4, 4, 1], dtype=tf.int32)
    
    def call(self, inputs):
        # inputs is out_backprop actually based on the example usage
        # The depthwise_conv2d_backprop_input expects: input_sizes, filter, out_backprop
        # Inputs shape must be compatible with expected out_backprop shape
        
        # Defensive type & shape check
        if not isinstance(inputs, tf.Tensor):
            raise TypeError("inputs must be a tf.Tensor")

        # Perform the depthwise_conv2d_backprop_input:
        # Note: tf.compat.v2.nn.depthwise_conv2d_backprop_input signature:
        # depthwise_conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding)
        # input_sizes: 1-D int32 vector [batch, in_height, in_width, in_channels]
        # filter: 4-D tensor [filter_height, filter_width, in_channels, channel_multiplier]
        # out_backprop: gradients w.r.t the output of forward depthwise conv (shape: [batch, out_height, out_width, in_channels * channel_multiplier])

        # Use TF function from tf.compat.v2.nn for compatibility with 2.20.0-dev nightly builds
        return tf.compat.v2.nn.depthwise_conv2d_backprop_input(
            input_sizes=self.input_sizes,
            filter=self.filter,
            out_backprop=inputs,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Based on example out_backprop shape:
    # if input_sizes = [1,4,4,1], filter = [1,1,1,1], strides=1, padding='VALID'
    # Output shape of forward depthwise conv would be [1, 4, 4, 1], so out_backprop shape = same
    # Provide a compatible random tensor out_backprop of shape [1,4,4,1], type float32
    return tf.random.uniform(shape=(1, 4, 4, 1), dtype=tf.float32)


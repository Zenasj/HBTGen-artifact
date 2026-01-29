# tf.random.normal((2, 1, 2, 2)) ← Input shape as used in the reproducer examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Kernel shape (2,2,2,1) matches conv filter of height=2, width=2,
        # input channels=2, output channels=1 as in the reproducer
        self.kernel = tf.Variable(tf.random.normal((2, 2, 2, 1)), trainable=True)

    @tf.function
    def call(self, x):
        # We perform a Conv2D with VALID padding and stride 1 (no dilation)
        # which on very small inputs can produce empty output shapes,
        # causing ValueErrors during shape inference in graph mode.
        #
        # To reconcile eager and graph mode behavior:
        # - We try to mimic the eager mode behavior that returns empty tensors instead of errors.
        # - We add explicit shape inference guard to allow empty spatial output.
        #
        # Strategy here: perform shape checks before calling raw_ops.Conv2D.
        # If the output spatial dimension would be negative (i.e. no valid conv),
        # return an explicit empty tensor with correct dtype and shape zero in spatial dims.
        #
        # This aims to behave consistently on both eager and graph modes,
        # for use in tf.function with jit_compile=True (XLA compatible).
        
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        in_channels = input_shape[3]

        # Extract kernel spatial shape
        kernel_height = tf.shape(self.kernel)[0]
        kernel_width = tf.shape(self.kernel)[1]
        out_channels = tf.shape(self.kernel)[3]

        # Compute output spatial dimensions for VALID padding:
        # output_height = floor((height - kernel_height)/stride) + 1
        # stride is 1 from source code.
        out_height = height - kernel_height + 1
        out_width = width - kernel_width + 1

        # If either output dimension is <=0, return empty tensor with shape (batch, 0, 0, out_channels)
        def empty_output():
            empty_shape = tf.stack([batch_size, 0, 0, out_channels])
            return tf.zeros(empty_shape, dtype=x.dtype)

        def conv_output():
            return tf.raw_ops.Conv2D(
                input=x,
                filter=self.kernel,
                strides=[1, 1, 1, 1],
                padding='VALID',
                data_format='NHWC')

        return tf.cond(
            tf.logical_or(out_height <= 0, out_width <= 0),
            empty_output,
            conv_output)

def my_model_function():
    return MyModel()

def GetInput():
    # The input shape used in examples is [2, 1, 2, 2], which triggers the edge case.
    # This shape is batch=2, height=1, width=2, channels=2 in NHWC format.
    # However, the examples used channels_last format NHWC but with channels=2 in kernel input,
    # input channels dimension in input is 1 in reproducer, but filter expects 2.
    #
    # To make shapes consistent for Conv2D:
    # Input: [batch, height, width, channels]
    # Kernel: [kernel_height, kernel_width, in_channels, out_channels]
    # Kernel in_channels=2 means input should have channels=2.
    #
    # The reproducer generates input x shape (2, 1, 2, 2) which is NHWC (batch, height, width, channels)
    return tf.random.normal(shape=(2, 1, 2, 2), dtype=tf.float32)


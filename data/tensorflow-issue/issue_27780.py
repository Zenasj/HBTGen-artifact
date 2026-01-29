# tf.random.uniform((B, 224, 224, 3), dtype=tf.float16 or tf.float32) ← This matches the FakeImageDataInput image_size 224 and 3 channels

import numpy as np
import tensorflow as tf

# Kernel initializers based on the issue's code
def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)

class TestBlock(tf.keras.layers.Layer):
    """
    One building block used in the reported model.
    Supports either DepthwiseConv2D or standard Conv2D for the middle convolution.
    """

    def __init__(self, kernel_size, input_filters, output_filters,
                 stride=1, use_depthwise_conv=True, **kwargs):
        super(TestBlock, self).__init__(**kwargs)
        self._use_depthwise_conv = use_depthwise_conv
        self._kernel_size = kernel_size
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._stride = stride

        self._expand_conv = tf.keras.layers.Conv2D(
            filters=self._input_filters * 6,
            kernel_size=(1, 1),
            strides=(self._stride, self._stride),
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        self._bn0 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=1e-3, fused=True)

        if self._use_depthwise_conv:
            self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self._kernel_size,
                strides=(self._stride, self._stride),
                depthwise_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
        else:
            self._depthwise_conv = tf.keras.layers.Conv2D(
                filters=self._input_filters * 6,
                kernel_size=self._kernel_size,
                strides=(self._stride, self._stride),
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
        self._bn1 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=1e-3, fused=True)

        self._project_conv = tf.keras.layers.Conv2D(
            filters=self._output_filters,
            kernel_size=(1, 1),
            strides=(self._stride, self._stride),
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        self._bn2 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=1e-3, fused=True)

    def call(self, inputs, training=True):
        x = self._expand_conv(inputs)
        x = self._bn0(x, training=training)
        x = tf.nn.relu(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self._project_conv(x)
        x = self._bn2(x, training=training)
        return x


class MyModel(tf.keras.Model):
    """
    The model fuses two paths:
    - One using DepthwiseConv2D
    - One using regular Conv2D instead of depthwise

    It then compares their outputs at the final logits layer by computing
    the absolute difference and returns this difference as output.

    This reflects the issue context where the author compared speed and output differences
    between depthwise conv and conv2d variants.
    """

    def __init__(self):
        super(MyModel, self).__init__()

        # Stem conv layers
        self._conv_stem_depthwise = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(2, 2),
            kernel_initializer=conv_kernel_initializer, padding='same', use_bias=False)
        self._bn0_depthwise = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, fused=True)

        self._conv_stem_conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(2, 2),
            kernel_initializer=conv_kernel_initializer, padding='same', use_bias=False)
        self._bn0_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, fused=True)

        # Blocks with depthwise conv
        self._blocks_depthwise = [
            TestBlock(kernel_size=(3,3), input_filters=32, output_filters=16, stride=1, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=16, output_filters=24, stride=1, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=24, output_filters=24, stride=2, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=24, output_filters=40, stride=1, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=40, stride=1, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=40, stride=2, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=80, stride=1, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=80, stride=2, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=80, output_filters=112, stride=1, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=112, output_filters=160, stride=2, use_depthwise_conv=True),
            TestBlock(kernel_size=(3,3), input_filters=160, output_filters=320, stride=1, use_depthwise_conv=True)
        ]
        self._bn_head_depthwise = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, fused=True)
        self._conv_head_depthwise = tf.keras.layers.Conv2D(
            filters=1280, kernel_size=(1,1), kernel_initializer=conv_kernel_initializer, padding='same', use_bias=False)
        self._pool_depthwise = tf.keras.layers.GlobalAveragePooling2D()
        self._fc_depthwise = tf.keras.layers.Dense(1000, kernel_initializer=dense_kernel_initializer)

        # Blocks with regular conv (no depthwise)
        self._blocks_conv = [
            TestBlock(kernel_size=(3,3), input_filters=32, output_filters=16, stride=1, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=16, output_filters=24, stride=1, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=24, output_filters=24, stride=2, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=24, output_filters=40, stride=1, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=40, stride=1, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=40, stride=2, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=80, stride=1, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=40, output_filters=80, stride=2, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=80, output_filters=112, stride=1, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=112, output_filters=160, stride=2, use_depthwise_conv=False),
            TestBlock(kernel_size=(3,3), input_filters=160, output_filters=320, stride=1, use_depthwise_conv=False)
        ]
        self._bn_head_conv = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, fused=True)
        self._conv_head_conv = tf.keras.layers.Conv2D(
            filters=1280, kernel_size=(1,1), kernel_initializer=conv_kernel_initializer, padding='same', use_bias=False)
        self._pool_conv = tf.keras.layers.GlobalAveragePooling2D()
        self._fc_conv = tf.keras.layers.Dense(1000, kernel_initializer=dense_kernel_initializer)

    def call(self, inputs, training=True):
        # Depthwise conv branch
        x_dw = self._conv_stem_depthwise(inputs)
        x_dw = self._bn0_depthwise(x_dw, training=training)
        x_dw = tf.nn.relu(x_dw)
        for block in self._blocks_depthwise:
            x_dw = block(x_dw, training=training)
        x_dw = self._conv_head_depthwise(x_dw)
        x_dw = self._bn_head_depthwise(x_dw, training=training)
        x_dw = tf.nn.relu(x_dw)
        x_dw = self._pool_depthwise(x_dw)
        logits_dw = self._fc_depthwise(x_dw)

        # Conv2D branch
        x_conv = self._conv_stem_conv(inputs)
        x_conv = self._bn0_conv(x_conv, training=training)
        x_conv = tf.nn.relu(x_conv)
        for block in self._blocks_conv:
            x_conv = block(x_conv, training=training)
        x_conv = self._conv_head_conv(x_conv)
        x_conv = self._bn_head_conv(x_conv, training=training)
        x_conv = tf.nn.relu(x_conv)
        x_conv = self._pool_conv(x_conv)
        logits_conv = self._fc_conv(x_conv)

        # Compute absolute difference between outputs
        diff = tf.abs(logits_dw - logits_conv)

        # Optionally, return all three for inspection
        return diff, logits_dw, logits_conv

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a single batch input tensor of shape [batch_size, height, width, channels]
    batch_size = 8  # smaller batch for fast test, can be adjusted
    height = 224
    width = 224
    channels = 3

    # Based on the reported usage: input uses float16 for mixed precision or float32 otherwise
    # We default here to float16 to reflect mixed precision scenario in the issue.
    dtype = tf.float16

    # Use uniform random values as a placeholder input matching the expected shape
    return tf.random.uniform(shape=(batch_size, height, width, channels), dtype=dtype)


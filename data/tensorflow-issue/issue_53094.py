# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← assuming input shape typical for MobileNetV3

import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend

# Helper functions inferred from MobileNetV3 style implementations

def _depth(d):
    # Ensures divisibility by 8
    return max(int(d + 7) // 8 * 8, 8)

def _inverted_res_block(inputs, expansion, filters, kernel, stride, activation, block_id, kernel_regularizer=None):
    channel_axis = -1
    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = _depth(filters)
    x = inputs
    prefix = f'expanded_conv_{block_id}_' if block_id is not None else 'expanded_conv_'

    # 1x1 expansion convolution
    if expansion != 1:
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          kernel_regularizer=kernel_regularizer,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand/BatchNorm')(x)
        x = activation(x)
    else:
        expansion = 1

    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=kernel,
                               strides=stride,
                               padding='same',
                               use_bias=False,
                               depthwise_regularizer=kernel_regularizer,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(x)

    # Project convolution (pointwise)
    x = layers.Conv2D(pointwise_conv_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=kernel_regularizer,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and in_channels == pointwise_conv_filters:
        x = layers.Add(name=prefix + 'add')([inputs, x])
    return x

def relu(x):
    return tf.nn.relu(x)

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224), 
                 output_bias=None,
                 l1_factor=0,
                 l2_factor=0,
                 dropout_rate=0.2,
                 alpha=1.0):
        super().__init__()
        # Setup kernel regularizer as in original code
        if l1_factor > 0 and l2_factor == 0:
            kernel_regularizer = regularizers.L1(l1_factor)
        elif l2_factor > 0 and l1_factor == 0:
            kernel_regularizer = regularizers.L2(l2_factor)
        elif l1_factor > 0 and l2_factor > 0:
            kernel_regularizer = regularizers.L1L2(l1_factor, l2_factor)
        else:
            kernel_regularizer = None

        self.alpha = alpha
        self.channel_axis = -1
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate

        # Building layers from the original build_model function (converted for subclassing)
        self.conv = layers.Conv2D(16,
                                  kernel_size=3,
                                  strides=(2, 2),
                                  padding='same',
                                  use_bias=False,
                                  kernel_regularizer=kernel_regularizer,
                                  name='Conv')
        self.bn_conv = layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm')
        self.activation = relu

        # Depth helper
        def depth(d):
            return _depth(d * alpha)

        # Inverted residual blocks (parameters from the snippet)
        # These will be stored as list/layers for forward call
        self.irb_0 = lambda x: _inverted_res_block(x, 1, depth(16), 3, 2, relu, 0, kernel_regularizer)
        self.irb_1 = lambda x: _inverted_res_block(x, 72. / 16, depth(24), 3, 2, relu, 1, kernel_regularizer)
        self.irb_2 = lambda x: _inverted_res_block(x, 88. / 24, depth(24), 3, 1, relu, 2, kernel_regularizer)
        self.irb_3 = lambda x: _inverted_res_block(x, 4, depth(40), 3, 2, relu, 3, kernel_regularizer)
        self.irb_4 = lambda x: _inverted_res_block(x, 6, depth(40), 3, 1, relu, 4, kernel_regularizer)
        self.irb_5 = lambda x: _inverted_res_block(x, 6, depth(40), 3, 1, relu, 5, kernel_regularizer)
        self.irb_6 = lambda x: _inverted_res_block(x, 6, depth(96), 3, 2, relu, 8, kernel_regularizer)

        # Last conv layers
        # last_conv_ch = _depth(backend.int_shape(x)[channel_axis]*6) --- need to delay calculation until call
        self.conv_1 = layers.Conv2D(
            filters=None,  # placeholder, set dynamically below
            kernel_size=1,
            padding='same',
            use_bias=False,
            name='Conv_1'
        )
        self.bn_conv_1 = layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1/BatchNorm')
        self.avg_pool = layers.AveragePooling2D(pool_size=7, strides=1)
        self.conv_2 = layers.Conv2D(
            filters=128,  # last_point_ch = 128 as per code snippet (not multiplied further)
            kernel_size=1,
            padding='same',
            use_bias=True,
            name='Conv_2'
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        self.logits_conv = layers.Conv2D(
            filters=2,
            kernel_size=1,
            padding='same',
            name='Logits'
        )
        self.flatten = layers.Flatten()
        self.softmax = layers.Softmax()

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x)
        x = self.bn_conv(x, training=training)
        x = self.activation(x)

        x = self.irb_0(x)
        x = self.irb_1(x)
        x = self.irb_2(x)
        x = self.irb_3(x)
        x = self.irb_4(x)
        x = self.irb_5(x)
        x = self.irb_6(x)

        # Calculate last_conv_ch dynamically as in original code
        last_conv_ch = _depth(backend.int_shape(x)[self.channel_axis] * 6)
        if self.alpha > 1.0:
            last_point_ch = _depth(128 * self.alpha)
        else:
            last_point_ch = 128

        # Redefine conv_1 layer dynamically if shapes differ (workaround since filters depend on last_conv_ch)
        # As layer parameters must be defined at init, we recreate conv_1 and conv_2 dynamically for new filters:

        # Note: This will slow down first call due to layer recreation, but is necessary due to dynamic shape
        if self.conv_1.filters != last_conv_ch:
            self.conv_1 = layers.Conv2D(last_conv_ch,
                                        kernel_size=1,
                                        padding='same',
                                        use_bias=False,
                                        name='Conv_1')
            self.bn_conv_1 = layers.BatchNormalization(axis=self.channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1/BatchNorm')

        if self.conv_2.filters != last_point_ch:
            self.conv_2 = layers.Conv2D(last_point_ch,
                                        kernel_size=1,
                                        padding='same',
                                        use_bias=True,
                                        name='Conv_2')

        x = self.conv_1(x)
        x = self.bn_conv_1(x, training=training)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.conv_2(x)
        x = self.activation(x)

        if self.dropout_rate > 0 and training:
            x = self.dropout(x, training=training)

        x = self.logits_conv(x)
        x = self.flatten(x)
        prob = self.softmax(x)
        return prob


def my_model_function():
    # Return instance of MyModel with typical input shape of 224x224, RGB images, matching MobileNetV3 style
    return MyModel(input_shape=(224, 224), dropout_rate=0.2, alpha=1.0)


def GetInput():
    # Return a random input tensor matching expected input shape: batch size 1, height 224, width 224, 3 channels (RGB)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)


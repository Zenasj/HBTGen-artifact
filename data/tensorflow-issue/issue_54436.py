from tensorflow.keras import layers

def call(self, x, training=False):
    for block in self.blocks:
        h = x
        for conv in block:
            h = conv(h, training=training)

        x = x + h

    return x

def call(self, x, training=False):
    for block, add in zip(self.blocks, self.adds):
        h = x
        for conv in block:
            h = conv(h, training=training)

        x = add([x, h])

    return x

from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow.nn
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow import keras

def plot_model(model, shape):
    inputs = keras.Input(shape[1:])
    ones = tf.ones(shape)
    model(ones)  # I think needed to properly init graph for plotting
    outputs = model.call(inputs)
    wrapped_model = keras.Model(inputs, outputs)
    return tensorflow.keras.utils.plot_model(
        wrapped_model, expand_nested=True, show_shapes=True)

class ConvBnAct(L.Layer):

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        activation: Optional[str] = 'swish',
        use_bias=False,
        use_batch_norm=True,
        data_format='channels_last'
            ):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias
        self.data_format = data_format

        self.activation = L.Activation(activation)
        self.act_type = activation

        bn_axis = 1 if data_format == 'channels_first' else -1
        self.batch_norm = L.BatchNormalization(
            axis=bn_axis) if use_batch_norm else None

    def build(self, input_shape):
        self.conv = L.Conv2D(
            self.out_channels,
            self.kernel_size,
            input_shape=input_shape[1:],
            padding='same',
            strides=self.stride,
            activation=None,
            use_bias=self.use_bias,
            data_format=self.data_format,
            )

    def call(self, inputs, training=False):
        x = self.conv(inputs)

        if self.batch_norm:
            x = self.batch_norm(x, training=training)

        if self.activation:
            x = self.activation(x)

        return x

class ResBlock(L.Layer):

    def __init__(
        self,
        blocks: int,
        shortcut=True,
        data_format='channels_last'
    ):
        super().__init__()
        self.n_blocks = blocks
        self.shortcut = shortcut
        self.data_format = data_format

    def build(self, input_shape):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        channels = input_shape[channel_axis]

        self.blocks = []
        for i in range(self.n_blocks):
            block = [
                ConvBnAct(channels, kernel_size=1, stride=1, data_format=self.data_format),
                ConvBnAct(channels, kernel_size=3, stride=1, data_format=self.data_format)
                ]
            self.blocks.append(block)

    def call(self, x, training=False):
        for block in self.blocks:
            h = x
            for conv in block:
                h = conv(h, training=training)

            x = x + h if self.shortcut else h

        return x

if __name__ == '__main__':
    i = keras.Input((24, 24, 3))
    r = ResBlock(2, True)
    plot_model(r, (1, 24, 24, 3))

class ConvBnAct(object):

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        activation: Optional[str] = 'swish',
        use_bias=False,
        use_batch_norm=True,
        data_format='channels_last'
            ):
        super().__init__()
        self.kernel = tf.ones((kernel_size, kernel_size, 3, 3))

    def __call__(self, inputs, training=False):
        x = tf.nn.conv2d(inputs, self.kernel, strides=1, padding='SAME')
        return x

class ResBlock(L.Layer):

    def __init__(self, blocks: int, data_format='channels_last'):
        super().__init__()
        self.n_blocks = blocks

        self.blocks = []
        for i in range(self.n_blocks):
            block = [
                ConvBnAct(3, kernel_size=1, stride=1, data_format=self.data_format),
                ConvBnAct(3, kernel_size=3, stride=1, data_format=self.data_format)
                ]
            self.blocks.append(block)

    def call(self, x, training=False):
        for block in self.blocks:
            h = x
            for conv in block:
                h = conv(h, training=training)

            x = x + h

        return x

class ResBlock(object):

    def __init__(self, blocks: int, data_format='channels_last'):
        super().__init__()
        self.n_blocks = blocks

        self.blocks = []
        for i in range(self.n_blocks):
            block = [
                ConvBnAct(3, kernel_size=1, stride=1, data_format=data_format),
                ConvBnAct(3, kernel_size=3, stride=1, data_format=data_format)
                ]
            self.blocks.append(block)

    def __call__(self, x, training=False):
        for block in self.blocks:
            h = x
            for conv in block:
                h = conv(h, training=training)

            x = x + h

        return x
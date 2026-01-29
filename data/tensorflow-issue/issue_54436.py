# tf.random.uniform((B, H, W, C), dtype=tf.float32)  # Assuming typical image input format, e.g. (batch, 24, 24, 3)

import tensorflow as tf
from tensorflow.keras import layers as L

class ConvBnAct(L.Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride,
        activation='swish',
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

        self.activation = L.Activation(activation) if activation else None
        self.act_type = activation

        bn_axis = 1 if data_format == 'channels_first' else -1
        self.batch_norm = L.BatchNormalization(axis=bn_axis) if use_batch_norm else None

    def build(self, input_shape):
        self.conv = L.Conv2D(
            self.out_channels,
            self.kernel_size,
            strides=self.stride,
            padding='same',
            activation=None,
            use_bias=self.use_bias,
            data_format=self.data_format,
            input_shape=input_shape[1:],
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(L.Layer):
    """Residual block composed of blocks of ConvBnAct layers with skip connections.

    Uses explicit keras.layers.Add to avoid incorrect graph issues with `+` operator.
    """
    def __init__(self, blocks: int, shortcut=True, data_format='channels_last'):
        super().__init__()
        self.n_blocks = blocks
        self.shortcut = shortcut
        self.data_format = data_format
        self.adds = []  # List of keras.layers.Add instances for skip connections

    def build(self, input_shape):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        channels = input_shape[channel_axis]

        self.blocks = []
        self.adds = []
        for _ in range(self.n_blocks):
            block = [
                ConvBnAct(channels, kernel_size=1, stride=1, data_format=self.data_format),
                ConvBnAct(channels, kernel_size=3, stride=1, data_format=self.data_format)
            ]
            self.blocks.append(block)
            self.adds.append(L.Add())  # One Add layer per residual addition

        super().build(input_shape)

    def call(self, x, training=False):
        for block, add in zip(self.blocks, self.adds):
            h = x
            for conv in block:
                h = conv(h, training=training)

            if self.shortcut:
                # Use explicit Add layer to create correct graph with skip connection
                x = add([x, h])
            else:
                x = h
        return x


class MyModel(tf.keras.Model):
    """Fused model demonstrating the correct use of Residual blocks with explicit Add layers."""

    def __init__(self, input_shape=(24,24,3), blocks=2, data_format='channels_last'):
        super().__init__()
        self.data_format = data_format
        self.input_shape_ = input_shape

        # One ResBlock instance, you can add more layers or blocks as needed
        self.resblock = ResBlock(blocks=blocks, shortcut=True, data_format=data_format)

    def call(self, inputs, training=False):
        x = inputs
        x = self.resblock(x, training=training)
        return x


def my_model_function():
    # Instantiate MyModel with a default input shape matching the original example
    model = MyModel(input_shape=(24, 24, 3), blocks=2)
    # Build the model by calling on a dummy input to initialize weights
    dummy_input = tf.random.uniform((1, 24, 24, 3), dtype=tf.float32)
    model(dummy_input)
    return model


def GetInput():
    # Return a random input tensor matching the expected input for MyModel
    # Assumes batch size 1, height 24, width 24, channels 3 for typical image input
    return tf.random.uniform((1, 24, 24, 3), dtype=tf.float32)


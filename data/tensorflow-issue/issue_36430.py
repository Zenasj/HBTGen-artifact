# tf.random.uniform((B, 241, 80), dtype=tf.float32) ← Input shape inferred from given input tensor shape

import numpy as np
import tensorflow as tf

class TFReflectionPad1d(tf.keras.layers.Layer):
    def __init__(self, padding_size):
        super(TFReflectionPad1d, self).__init__()
        self.padding_size = padding_size

    def call(self, x):
        # Pad the 2nd dim (time dimension) with reflection padding
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], "REFLECT")

class TFUpsampleConv1d(tf.keras.layers.Layer):
    def __init__(self, upsample_factor, filters, kernel_size, padding='same'):
        super(TFUpsampleConv1d, self).__init__()
        self.upsample1d = tf.keras.layers.UpSampling1D(size=upsample_factor)
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding=padding)

    def call(self, x):
        x = self.upsample1d(x)
        return self.conv1d(x)

class TFResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, dilation=1):
        super(TFResnetBlock, self).__init__()
        # A ResNet block with LeakyReLU, reflection pad with dilation, convolution, skip connection
        self.block = [
            tf.keras.layers.LeakyReLU(0.2),
            TFReflectionPad1d(dilation),
            tf.keras.layers.Conv1D(filters=dim, kernel_size=3, dilation_rate=dilation),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(filters=dim, kernel_size=1),
        ]
        self.shortcut = tf.keras.layers.Conv1D(filters=dim, kernel_size=1)

    def call(self, x):
        _x = tf.identity(x)
        for layer in self.block:
            _x = layer(_x)
        return self.shortcut(x) + _x

class MyModel(tf.keras.Model):
    def __init__(self, ngf=32, n_residual_layers=3):
        super(MyModel, self).__init__()
        # Inspired by TFMelGANGenerator with reflection padding, convolutions and upsampling
        # This matches the model from the issue, encapsulated in MyModel

        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model_layers = [
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(filters=mult * ngf, kernel_size=7, padding='valid')
        ]

        for i, r in enumerate(ratios):
            model_layers += [
                tf.keras.layers.LeakyReLU(0.2),
                TFUpsampleConv1d(
                    upsample_factor=r,
                    filters=mult * ngf // 2,
                    kernel_size=r * 2 - 1,
                    padding='same'
                )
            ]

            for j in range(n_residual_layers):
                model_layers.append(TFResnetBlock(dim=mult * ngf // 2, dilation=3 ** j))

            mult //= 2

        model_layers += [
            tf.keras.layers.LeakyReLU(0.2),
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(filters=1, kernel_size=7, padding='valid'),
            tf.keras.layers.Activation('tanh')
        ]

        self.model = tf.keras.Sequential(model_layers)

    def call(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel with default ngf=32, n_residual_layers=3
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching the input expected by MyModel
    # Input shape: (batch_size, time_steps=241, channels=80)
    batch_size = 1  # Assuming batch size 1 for example
    time_steps = 241
    channels = 80
    return tf.random.uniform((batch_size, time_steps, channels), dtype=tf.float32)


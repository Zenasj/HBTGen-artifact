# tf.random.uniform((B, H, W, C), dtype=tf.float32) where B=32, H=90, W=180, C=1 (channels_first data format)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The issue example uses a Conv2D layer with channels_first data format
        # Filters=32, kernel_size=3x3, padding='same', stride=1
        # This is the only layer shown in the minimal repro snippet
        self.conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format='channels_first',
            name='upscale'
        )

    def call(self, inputs):
        # Input is expected to be NHWC or NCHW depending on data_format 'channels_first' means NCHW:
        # But as per the "GetInput" below, we generate NCHW tensors
        x = self.conv(inputs)
        return x


def my_model_function():
    return MyModel()


def GetInput():
    # According to the code snippet, the input to the model is shape:
    # Batch size 32, channels_first data format, dims 90 x 180 with 1 channel.
    # So input shape is (B=32, C=1, H=90, W=180)
    # We generate a random tensor with these dims and float32 dtype.

    # Assumption: Since conv layer expects 4D tensor with channels_first,
    # input shape: (batch, channels, height, width)
    batch_size = 32
    channels = 1
    height = 90
    width = 180
    input_tensor = tf.random.uniform(shape=(batch_size, channels, height, width), dtype=tf.float32)
    return input_tensor


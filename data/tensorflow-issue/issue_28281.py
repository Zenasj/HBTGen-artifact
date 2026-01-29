# tf.random.uniform((B, None, None, 3), dtype=tf.float32) ← Input shape is dynamic spatial dims with 3 channels

import tensorflow as tf

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Downsample stack
        self.down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample(128, 4),  # (bs, 64, 64, 128)
            downsample(256, 4),  # (bs, 32, 32, 256)
            downsample(512, 4),  # (bs, 16, 16, 512)
            downsample(512, 4),  # (bs, 8, 8, 512)
            downsample(512, 4),  # (bs, 4, 4, 512)
            downsample(512, 4),  # (bs, 2, 2, 512)
            downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        # Upsample stack
        self.up_stack = [
            upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4),  # (bs, 16, 16, 1024)
            upsample(256, 4),  # (bs, 32, 32, 512)
            upsample(128, 4),  # (bs, 64, 64, 256)
            upsample(64, 4),   # (bs, 128, 128, 128)
        ]

        self.last = tf.keras.layers.Conv2DTranspose(
            OUTPUT_CHANNELS, 4,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            activation='tanh')  # (bs, 256, 256, 3)

        self.concat = tf.keras.layers.Concatenate()

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = inputs

        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x, training=training)
            x = self.concat([x, skip])

        x = self.last(x)

        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor with shape (batch, height, width, channels)
    # We'll choose a batch size of 1 and spatial dims 256x256 (as pix2pix uses 256x256 images)
    batch_size = 1
    height = 256
    width = 256
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)


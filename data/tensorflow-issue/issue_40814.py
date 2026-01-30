import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    logits = get_logits()
    model = tf.keras.Model(inputs, logits)


model.fit(X, y)  # X and y are datasets read from tfrecords

import numpy as np
import tensorflow as tf


class Conv2dLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.layers.LeakyReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same", kernel_initializer="he_normal",
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        return config

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class UpSampleLayer(tf.keras.layers.Layer):
    def __init__(self, filters, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.activation = tf.keras.layers.LeakyReLU()
        self.upconv = tf.keras.layers.Conv2DTranspose(
            filters, 4, strides=strides, padding="same", kernel_initializer="he_normal"
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.filters = filters
        self.strides = strides

    def call(self, inputs, **kwargs):
        x = self.upconv(inputs)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return self.activation(x)

    def get_config(self):
        config = super().get_config()
        config["filters"] = self.filters
        config["strides"] = self.strides
        return config


class DownsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.conv1 = Conv2dLayer(filters, 4)
        self.conv2 = Conv2dLayer(filters, 4)
        self.downsample_conv = Conv2dLayer(filters, 4, strides=2)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.downsample_conv(x)
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config["filters"] = self.filters
        return config


class Unet(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = tf.keras.layers.Activation("relu")
        self.axis = -1
        self.downsample_blocks = []
        self.upsample_blocks = []

        n_maps_list = []

        for i in range(6):
            n_maps = 16 * 2 ** i
            n_maps_list.insert(0, n_maps)
            self.downsample_blocks.append(DownsampleBlock(n_maps))

        for i, n_maps in enumerate(n_maps_list[1:]):
            self.upsample_blocks.append(UpSampleLayer(n_maps, strides=2))
        self.upsample_blocks.append(UpSampleLayer(2, strides=2))

    def call(self, inputs, training=None, mask=None):
        skip_connections = []
        x = inputs
        for downsample_block in self.downsample_blocks:
            x = downsample_block(x)
            skip_connections.insert(0, x)

        x = self.upsample_blocks[0](x)  # no skip connection used for first block
        for upsample_block, h in zip(self.upsample_blocks[1:], skip_connections[1:]):
            x = upsample_block(tf.keras.layers.concatenate([x, h], axis=self.axis))
        return self.mask(x)


def train():
    BATCH_SIZE = 16
    WIDTH = 256
    HEIGHT = 512
    CHANNELS = 2
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Unet()
        model.build(input_shape=(None, WIDTH, HEIGHT, CHANNELS))
        model.compile(optimizer="adam", loss="mean_absolute_error")

    examples = np.random.rand(BATCH_SIZE * 20, WIDTH, HEIGHT, CHANNELS)
    target = np.random.rand(BATCH_SIZE * 20, WIDTH, HEIGHT, CHANNELS)

    ds = tf.data.Dataset.from_tensor_slices((examples, target))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    model.fit(ds, steps_per_epoch=1875, epochs=10)


train()
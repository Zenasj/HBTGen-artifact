from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import functools
from collections.abc import Iterable


# TODO Check for correctness of the model implementation
class Unit3D(tf.keras.layers.Layer):
    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn='relu',
                 use_batch_norm=True,
                 use_bias=False,
                 is_training=False,
                 name='unit_3d'):
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._is_training = is_training
        self._pipeline = []
        self._pipeline.append(tf.keras.layers.Conv3D(
            filters=self._output_channels,
            kernel_size=self._kernel_shape,
            strides=self._stride,
            padding='same',
            use_bias=self._use_bias,
            data_format='channels_first'
        )
        )
        if self._use_batch_norm:
            bn = tf.keras.layers.BatchNormalization(
                axis=1,
                fused=False,
            )
            bn = functools.partial(bn, training=self._is_training)
            self._pipeline.append(bn)

        if self._activation is not None:
            self._pipeline.append(tf.keras.layers.Activation(
                activation=self._activation
            )
            )

        print(isinstance(self._pipeline, Iterable))
        print(type(self._pipeline))
        self._pipeline = lambda x: functools.reduce(lambda f, g: g(f), self._pipeline, x)

    def call(self, input):
        return self._pipeline(input)

import tensorflow as tf
from nets.i3d import Unit3D

model = Unit3D(output_channels=64, kernel_shape=[7,7,7],
               is_training=True)

input = tf.keras.backend.random_uniform(shape=(1,3,64,224,224),
                                        dtype=tf.float32)
output = model(input)

import tensorflow as tf
from i3d import Unit3D

model = Unit3D(output_channels=64, kernel_shape=[7,7,7],
               is_training=True)

input = tf.keras.backend.random_uniform(shape=(1,3,64,224,224),
                                        dtype=tf.float32)
output = model(input)
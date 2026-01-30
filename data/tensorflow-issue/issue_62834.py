from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import math
from pprint import pprint
import tensorflow as tf
from keras.models import load_model


def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0 / max(1., fan_in)
    stddev = math.sqrt(scale)
    return tf.initializers.TruncatedNormal(mean=0.0, stddev=stddev)
def _conv3x3(in_channel, out_channel, stride=1):
    return tf.keras.layers.Conv2D(out_channel, kernel_size=3, strides=stride, padding='same',
                                  kernel_initializer=conv_variance_scaling_initializer(in_channel, out_channel, 3),
                                  use_bias=False)


def _conv1x1(in_channel, out_channel, stride=1):
    return tf.keras.layers.Conv2D(out_channel, kernel_size=1, strides=stride, padding='same',
                                  # kernel_initializer=conv_variance_scaling_initializer(in_channel, out_channel, 1),
                                  use_bias=False)


def _conv7x7(in_channel, out_channel, stride=1):
    return tf.keras.layers.Conv2D(out_channel, kernel_size=7, strides=stride, padding='same',
                                  kernel_initializer=conv_variance_scaling_initializer(in_channel, out_channel, 7)
                                  , use_bias=False)


def _bn(channel):
    return tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-4)


def _fc(in_channel, out_channel):
    return tf.keras.layers.Dense(out_channel, activation=None,
                                 kernel_initializer=tf.keras.initializers.HeUniform())


class ResidualBlock(tf.keras.Model):
    def __init__(self, in_channel, out_channel, stride=1, use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.se_block = se_block
        channel = self.out_channel // 4  # Assuming expansion factor is 4

        self.conv1 = _conv1x1(self.in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        if self.use_se and self.stride != 1:
            # Adjusted for TensorFlow; using Keras layers
            self.e2 = tf.keras.Sequential([
                _conv3x3(channel, channel, stride=1),
                _bn(channel),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
            ])
        else:
            self.conv2 = _conv3x3(channel, channel, stride=stride)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, self.out_channel, stride=1)
        self.bn3 = _bn(self.out_channel)

        self.relu = tf.keras.layers.ReLU()

        self.down_sample = False
        if stride != 1 or self.in_channel != self.out_channel:
            self.down_sample = True
            self.down_sample_layer = tf.keras.Sequential([
                _conv1x1(in_channel, out_channel, stride),
                _bn(out_channel)
            ])

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'in_channel': self.in_channel,
            'out_channel': self.out_channel,
            # include other arguments if there are any
        })
        return config

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        if self.use_se and self.stride != 1:
            out = self.e2(out, training=training)
        else:
            out = self.conv2(out)
            out = self.bn2(out, training=training)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.down_sample:
            identity = self.down_sample_layer(identity, training=training)

        out = out + identity
        out = self.relu(out)

        return out


# 加载模型h5文件
mmodel = load_model("/root/zmx/COMET-master/data/origin_models/resnet.h5",
                    custom_objects={'ResidualBlock': ResidualBlock})
mmodel.summary()
model_json = mmodel.to_json()
pprint(model_json)
f = open("issue1.json", "w")
f.write(model_json)
f.close()
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class ConvBlock:
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        K=1,
        backbone="residual",
    ):
        super(ConvBlock, self).__init__()

        self.f = tf.keras.layers.Conv2D(
            c_out,
            kernel_size=kernel_size,
            strides=stride,
            padding="SAME",
            use_bias=bias,
            activation=None,
        )
        self.g = tf.keras.layers.Conv2D(
            c_out,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            activation=None,
        )
        self.K = K
        self.backbone = backbone

        self.bn_out = tf.keras.layers.BatchNormalization()
        self.bn_f1 = tf.keras.layers.BatchNormalization()
        self.c_out = c_out

    def __call__(self, x):
        f = self.f(tf.keras.layers.Activation("relu")(self.bn_f1(x)))
        h = f


        bn_g = tf.keras.layers.BatchNormalization()
        h = self.g(tf.keras.layers.Activation("relu")(bn_g(h)))

        return h


def net(backbone="cnn", K=5):

    inp = tf.keras.Input((28, 28, 1))

    conv = ConvBlock(1, 1, 3, K=K, backbone=backbone)
    x = conv(inp)
    f1 = x
    x = tf.keras.layers.Activation("relu")(x)

    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
    x = avg_pool(x)
    f2 = x
    flatten = tf.keras.layers.Flatten()
    x = flatten(x)

    fc = tf.keras.layers.Dense(10)
    out = fc(x)

    return tf.keras.Model(inp, [out, f1, f2])

if __name__=="__main__":
    import numpy as np
    model = net(backbone="cnn")
    model.compile()
    x = np.zeros((28, 28, 1))
    print(model(x))

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import logging
import tensorflow as tf

logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
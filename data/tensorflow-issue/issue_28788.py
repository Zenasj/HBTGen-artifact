from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf # TF2

class Conv2D_BN_ReLU(tf.keras.Model):
    """Conv2D + BN + ReLU"""
    def __init__(self,
            filters,
            kernel_size,
            strides=1,
            padding="SAME",
            dilation_rate=1,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=None,
            **bn_params):
        super(Conv2D_BN_ReLU, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                padding=padding, dilation_rate=dilation_rate, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.bn = tf.keras.layers.BatchNormalization(**bn_params)

    def call(self, x, training=None):
        x = self.conv(x)
        x = tf.nn.relu(self.bn(x, training=training))
        return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    net = Conv2D_BN_ReLU(64, 1, 1, dilation_rate=6)
    x = tf.ones((1, 32, 32, 64))
    y = net(x)

    x = tf.ones((1, 64, 64, 64))
    y = net(x)

class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b
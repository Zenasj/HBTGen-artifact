from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

class SNetwork(tf.keras.Model):
    def __init__(self):
        super(SNetwork, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(8, 8),
                                            strides=(4, 4),
                                            padding="same",
                                            input_shape=(96, 96, 3),
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation="linear")

        self.flatten = tf.keras.layers.Flatten()

        self.a_dense = tf.keras.layers.Dense(512,
                                             kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                             bias_initializer=tf.keras.initializers.Zeros(),
                                             activation="relu")

        self.a_out = tf.keras.layers.Dense(9,
                                           bias_initializer=tf.keras.initializers.Zeros(),
                                           activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)
        x = self.a_dense(x)
        x = self.a_out(x)
        return x

tf.keras.backend.set_floatx("float16")
tmp = np.zeros((1, 96, 96, 3), dtype="float16")
net = SNetwork()
print(net(tmp))
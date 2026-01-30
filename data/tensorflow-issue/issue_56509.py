import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units

        self.projection = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training):
        x = self.projection(inputs[:, :, :, 3:])

        A = inputs[:, :, :, 2]
        x = x + self.positional_encoding(A, self.units)

        B = inputs[:, :, :, 0]
        C = inputs[:, :, :, 1]
        x = x + self.positional_encoding(B, self.units)
        x = x + self.positional_encoding(C, self.units)

        return self.dropout(x, training=training)

    def positional_encoding(self, position, d_model, n=10000):
        angle_rads = position[:, :, :, tf.newaxis] / (
            tf.math.pow(n, (
                2 * (
                    tf.range(d_model, dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :]//2
                )
            ) / d_model)
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = tf.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = tf.cos(angle_rads[:, 1::2])

        return angle_rads

x = PositionalEmbedding(32, 0.1)
print(x(tf.random.normal((8, 7, 15, 4))))
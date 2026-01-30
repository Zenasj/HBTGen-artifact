import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.layers import *


class foo(tf.keras.Model):
    def __init__(self, rnn_units, dense_units, **kwargs):
        super().__init__(**kwargs)
        self.r1 = SimpleRNN(rnn_units)
        self.r2 = SimpleRNN(rnn_units)
        self.flat = tf.keras.layers.Flatten()
        self.d1 = Dense(rnn_units)
        self.d2 = Dense(dense_units)

    def call(self, inputs, **kwargs):

        x = self.r1(inputs)
        state = self.d1(self.flat(x))
        x = self.r2([inputs, state])
        x = self.d2(x)

        return x


train_input = tf.random.normal(shape=(6, 5, 10))
train_target = tf.random.normal(shape=(6, 8))

a = foo(10, 8)
a.compile(tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.MeanSquaredError())
a.fit(train_input, train_target)


b = SimpleRNN(10)(train_input)
state = Dense(10)(tf.reshape(b, (tf.shape(b)[0], -1)))
b = SimpleRNN(10)([train_input, state])
import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM

# Config
N = 32
T = 10
n = 2
mask_value = -1.
tf.random.set_seed(1)
np.random.seed(1)

# Data creation
X = np.ones((N, T, n)) * mask_value
Y = np.ones((N, T, 1)) * mask_value
for i in range(N):
    l = np.random.randint(1, T)
    value = np.random.random([l, n])
    X[i, :l] = value
    Y[i, :l] = np.array([sum(v) > 0.5 * n for v in value])[:, None]


class MyModel(Model):
    def __init__(self, n, mask_value, *args, **kwargs):
        super().__init__(name='MyModel', *args, **kwargs)
        self.mask_value = mask_value
        self.n = n
        self.LSTM = LSTM(self.n, return_sequences=True, activation='linear')
        return

    def call(self, inputs, training=None, mask=None):
        mask = tf.cast(tf.reduce_sum(inputs - self.mask_value, axis=-1), tf.bool)
        x = self.LSTM(inputs, mask=mask)
        return x


model = MyModel(n, mask_value)
model.build(input_shape=(N, T, n))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
model.summary()

mask = 1 - tf.cast(tf.reduce_all(tf.equal(X, mask_value), axis=-1), tf.float32)
loss_unmasked = tf.reduce_mean(tf.keras.losses.binary_crossentropy(Y, model.predict(X)))
loss_masked_1 = tf.reduce_sum(tf.keras.losses.binary_crossentropy(Y, model.predict(X)) * mask) / tf.reduce_sum(mask)
loss_masked_2 = tf.reduce_sum(tf.keras.losses.binary_crossentropy(Y, model.predict(X)) * mask) / (N * T)
print(f"model.evaluate(X, Y): {model.evaluate(X, Y)[0]:.2f}\n"
      f"loss_unmasked       : {loss_unmasked:.2f}\n"
      f"loss_masked_1       : {loss_masked_1:.2f}\n"
      f"loss_masked_2       : {loss_masked_2:.2f}"
      )
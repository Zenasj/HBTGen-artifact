from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


class TestSequence(tf.keras.utils.Sequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return batch_x, batch_y
    

seq = TestSequence(np.ones(10), np.ones(10), 2)

x_in = tf.keras.layers.Input((1,))
x_out = tf.keras.layers.Dense(1)(x_in)
model = tf.keras.Model(x_in, x_out)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(seq)
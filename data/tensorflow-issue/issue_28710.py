import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import math

import numpy as np
import tensorflow as tf


class SomeFeeder(tf.keras.utils.Sequence):
    """A dummy Sequence.

    Slightly modified tf.keras.utils.Sequence example.
    """

    def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


# Simple dummy model.
input_x = tf.keras.Input((4,))
output = tf.keras.layers.Dense(1, activation='sigmoid')(input_x)
model = tf.keras.Model(inputs=[input_x], outputs=[output])
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='binary_crossentropy')

# Dummy data.
x = np.random.rand(100, 4).astype(np.float32)
y = np.random.choice([0, 1], size=(100,)).astype(np.float32)
x = SomeFeeder(x, y, batch_size=4)

# Train the model with `steps_per_epoch=5`.
model.fit(x=x, epochs=10, steps_per_epoch=5)
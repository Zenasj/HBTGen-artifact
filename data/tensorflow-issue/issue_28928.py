import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

x = np.random.random((1000, 2))
y = 2 * x[:, 0] + 3 * x[:, 1] + np.random.normal(size=(1000,))

input_ = tf.keras.Input((2,), name="input")
output = tf.keras.layers.Dense(1)(input_)
model = tf.keras.Model(inputs=[input_], outputs=[output])

model.compile(tf.optimizers.Adam(), loss=tf.losses.mse)

model.fit(
    x, y, steps_per_epoch=2, batch_size=10,
)

model.fit(
    {"input": x}, y, steps_per_epoch=2, batch_size=10,
)

model.compile(
    tf.train.AdamOptimizer(),
    loss=tf.losses.mean_squared_error,
)
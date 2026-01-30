from tensorflow import keras
from tensorflow.keras import layers

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')
model = tf.keras.Sequential([ tf.keras.layers.Dense(units=1, input_shape=[1]) ])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=1)
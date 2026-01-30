import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()  # It works without this line

x, y = np.random.randn(100, 10), np.random.randn(100, 4)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(4, input_dim=10)])
model.compile(tf.train.RMSPropOptimizer(0.001), 'mse')

model.fit(x, y)  # Fitting without a generator works in eager mode

class Iterator:
    def __next__(self):
        return x, y

model.fit_generator(Iterator(), steps_per_epoch=10)
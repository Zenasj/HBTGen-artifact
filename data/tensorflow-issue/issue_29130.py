from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class MyExponentialUpdateLossScaleManager(ExponentialUpdateLossScaleManager):
    def variables(self):
        return [self._loss_scale, self._num_good_steps, self._num_bad_steps]

class MyLossScaleOptimizer(LossScaleOptimizer):
    def variables(self):
        return self._opt.variables() + self._loss_scale_manager.variables()

import numpy as np
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.contrib.mixed_precision import LossScaleOptimizer, ExponentialUpdateLossScaleManager

input = tf.keras.layers.Input(shape=(16,))
output = tf.keras.layers.Dense(1)(input)
model = tf.keras.models.Model(input, output)

optimizer = AdamOptimizer()
# Works without these two lines below
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000)
optimizer = LossScaleOptimizer(optimizer, loss_scale_manager)

model.compile(optimizer=optimizer, loss='binary_crossentropy')
model.fit(np.zeros((16, 16)), np.zeros((16,)))
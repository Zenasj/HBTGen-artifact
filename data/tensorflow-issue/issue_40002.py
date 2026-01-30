from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
print(tf.__version__)

model = tf.keras.Sequential([
    tf.keras.layers.Masking(1.),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.MeanSquaredError(),
              metrics=[tf.metrics.MeanSquaredError()],
              weighted_metrics=[tf.metrics.MeanSquaredError()])

print(model.train_on_batch(np.ones([1, 1]), np.ones([1, 1])))
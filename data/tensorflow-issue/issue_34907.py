from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.MeanSquaredError())

for i in range(1, 300):
    model.predict_on_batch(np.ones([i, 1]))
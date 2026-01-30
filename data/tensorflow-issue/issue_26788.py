from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

inputs = np.arange(10)
outputs = 2 * inputs

l2 = tf.keras.regularizers.L1L2(l2=0.0)
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(1, input_shape=[1], kernel_regularizer=l2)]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)
model.fit(inputs, outputs)
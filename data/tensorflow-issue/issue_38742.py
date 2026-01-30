import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = tf.keras.layers.Input([None])
summed = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
output = tf.keras.layers.Dense(1)(summed)
model = tf.keras.Model(inputs, output)
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanSquaredError())
for i in range(1, 300):
    model.train_on_batch(np.ones([1, i]), np.ones([1]) * i)

def dataset():
    for i in range(1, 300):
        yield np.ones([1, i]), np.ones([1]) * i

model.fit(dataset())
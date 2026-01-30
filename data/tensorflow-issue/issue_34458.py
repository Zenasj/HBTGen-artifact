from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices([1., 0., 2., 1.,])
dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))

train = tf.data.Dataset.zip((dataset, dataset)).batch(1).repeat().apply(tf.data.experimental.ignore_errors())
val = tf.data.Dataset.zip((dataset, dataset)).skip(2).batch(1).repeat().apply(tf.data.experimental.ignore_errors())

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
])
model.compile(optimizer='adam', loss='mse')
model.fit(train, epochs=10, steps_per_epoch=4, validation_data=val, validation_steps=2 )
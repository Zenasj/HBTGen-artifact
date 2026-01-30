from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

inp = x = tf.keras.Input(1)
x = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
x = tf.keras.layers.Activation(tf.nn.relu)(x)
model = tf.keras.Model(inp, x)

model.compile(loss="mse", optimizer="adam")
model.fit(
    tf.ones((32, 1)),
    tf.ones((32, 10)),
    callbacks=[tf.keras.callbacks.TensorBoard("logs", histogram_freq=1)],
    epochs=100,
)
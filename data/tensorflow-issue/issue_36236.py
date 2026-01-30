from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

inputs = tf.keras.layers.Input(shape=[None, 1], dtype=tf.float32)
hidden = tf.keras.layers.GRU(10)(inputs)
hidden = tf.gather(hidden, [0])
output = tf.keras.layers.Dense(1)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=output)

@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.losses.mean_squared_error(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

train(tf.constant([[[1], [2], [3]]], dtype=tf.float32), tf.constant([[1]], dtype=tf.float32))

inputs = tf.keras.layers.Input(shape=[None, 1], dtype=tf.float32)
hidden = tf.keras.layers.GRU(10)(inputs)
hidden = tf.gather(hidden * 1, [0])
output = tf.keras.layers.Dense(1)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=output)

@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.losses.mean_squared_error(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

train(tf.constant([[[1], [2], [3]]], dtype=tf.float32), tf.constant([[1]], dtype=tf.float32))

tensor_inputs.append(ops.convert_to_tensor(arg))

@tf.function
def summing_rnn(inputs):
  return tf.reduce_sum(inputs, axis=1)

@tf.function
def gradients(inputs):
  with tf.GradientTape() as tape:
    tape.watch(inputs)
    hidden = summing_rnn(inputs)
    hidden = tf.gather(hidden, tf.constant([0]))
    loss = tf.reduce_mean(hidden)
  return tape.gradient(loss, inputs)

gradients(tf.constant([[[1.0], [2.0]]])) # No error is raised
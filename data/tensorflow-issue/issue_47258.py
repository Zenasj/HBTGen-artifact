import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, batch_input_shape=(1, 384, 128), use_bias=True),
])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(2, batch_input_shape=(1, 384, 512)),
  tf.keras.layers.Dense(3)
])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, batch_input_shape=(1, 384, 128), use_bias=True),
])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(2, batch_input_shape=(1, 384, 512)),
  tf.keras.layers.Dense(3)
])
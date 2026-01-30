import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
  tf.keras.Input(shape=(512,), batch_size=1, dtype=tf.int32),
  tf.keras.layers.Embedding(1000, 128),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(5000),
  tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis=-1)),
])
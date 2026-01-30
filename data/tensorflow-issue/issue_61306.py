from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
gpus = ["/gpu:0", "/gpu:1", "/gpu:2"]
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(loss="mse", optimizer="sgd")
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(16)
model.fit(dataset)

import tensorflow as tf
gpus = ["/gpu:0", "/gpu:1", "/gpu:2"]
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss="mse", optimizer="sgd")
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(16,drop_remainder=True)
model.fit(dataset)
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

features = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
labels = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
train_dataset = tf.data.Dataset.zip((features, labels))

distribution = tf.contrib.distribute.MirroredStrategy()
with distribution.scope():
  inputs = tf.keras.layers.Input(shape=(1,))
  predictions = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
  model.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer(learning_rate=0.2))
model.fit(train_dataset, epochs=5, steps_per_epoch=10)
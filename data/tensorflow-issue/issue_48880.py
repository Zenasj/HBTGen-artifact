from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ["TPU_NAME"])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

class TestModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.embedding = tf.keras.layers.Embedding(input_dim=6, output_dim=64)
    self.pad_masking = tf.keras.layers.Masking(0, name="masking")
    lstm = tf.keras.layers.LSTM(64, return_sequences=True)
    self.lstm = tf.keras.layers.Bidirectional(lstm)
    self.dense = tf.keras.layers.Dense(1)

  def call(self, input, training=None):
    output = self.embedding(input)
    output = self.pad_masking(output)
    output = self.lstm(output)
    output = self.dense(output[:, -1, :])
    return output

with strategy.scope():
  x = tf.data.Dataset.from_tensor_slices([tf.constant([1,2,3,4,5]), tf.constant([1,2,3,4,5]), tf.constant([1,2,3,4,5])])
  y = tf.data.Dataset.from_tensor_slices([[1],[2],[3]])
  dataset = tf.data.Dataset.zip((x,y)).repeat().batch(2)

  model = TestModel()
  model(tf.keras.Input([None]))

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])
  model.fit(dataset, epochs=10, steps_per_epoch=10)
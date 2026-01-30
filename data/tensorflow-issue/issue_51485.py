import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf

print(f'tensorflow version {tf.__version__}')

n_batch = 1024
n_seq_len = 2048
n_bins = 64
n_nodes = 128

def create_dataset():
    ds = tf.data.Dataset.from_tensor_slices([tf.ones((n_batch, n_seq_len, n_bins))])
    def rand_batch(x):
        return tf.random.uniform((1,))*x, tf.random.uniform((1,))*x
    return ds.map(rand_batch).repeat()

def create_model():
  x = tf.keras.Input(shape=(n_seq_len, n_bins))
  g = tf.keras.layers.GRU(n_nodes, return_sequences=True)(x)
  g = tf.keras.layers.GRU(n_nodes, return_sequences=True)(g)
  g = tf.keras.layers.GRU(n_nodes, return_sequences=True)(g)
  g = tf.keras.layers.GRU(n_nodes, return_sequences=True)(g)
  # g = 1 * g # dummy OP
  g = tf.keras.layers.Dense(n_bins, activation='sigmoid')(g)
  return tf.keras.Model(inputs=[x], outputs=[g])

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

with strategy.scope():

  model = create_model()
  model.summary()
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
  model.fit(
      create_dataset(),
      steps_per_epoch=512,
      epochs=10,
      verbose=1
  )
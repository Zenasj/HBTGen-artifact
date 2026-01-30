from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import json
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import subprocess
import shlex
import sys

tfds.disable_progress_bar()

BUFFER_SIZE = 60000
BATCH_SIZE = 64

NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = NUM_WORKERS * BATCH_SIZE

if __name__ == "__main__":
  worker_addrs = ['localhost:9999', 'localhost:9998']
  os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': worker_addrs,
      },
      'task': {'type': 'worker', 'index': int(sys.argv[1])}
  })

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

  datasets, info = tfds.load(name='mnist',
                             with_info=True,
                             as_supervised=True)

  train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)

  train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)

  with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

   
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='/tmp/chk.hdf5',
        monitor='val_loss',
        save_best_only=True,
        load_weights_on_restart=True)

  multi_worker_model.fit(x=train_datasets, epochs=100, callbacks = [checkpoint_callback])

from datetime import datetime
from packaging import version
import os
import tensorflow as tf
import numpy as np
import json
# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '2048')
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["node67:12345", "node68:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})
def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size).prefetch(100)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(256, 2, activation='relu'),
      tf.keras.layers.Conv2D(128, 2, activation='relu'),
      tf.keras.layers.Conv2D(32, 1, activation='relu'),  
      tf.keras.layers.Conv2D(32, 2, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(2048, activation='relu'),        
      tf.keras.layers.Dense(1024, activation='relu'),        
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
    return model
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
num_workers = 2
per_worker_batch_size = 2048
# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=103, steps_per_epoch=70,callbacks = [tboard_callback])

if is_master:
  callbacks.add(tf.keras.callbacks.ModelCheckpoint(...))
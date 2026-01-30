import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import json
import os
import sys
import tensorflow as tf

#### if it is machine 1, machine 1 tfconfig 
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "chief": ["localhost:0000"],
        "worker": ["localhost:1111"]
    },
   "task": {"type": "chief", "index": 0}
})

#### else machine 2 tfconfig
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "chief": ["localhost:0000"],
        "worker": ["localhost:1111"]
    },
   "task": {"type": "worker", "index": 0}
})


strategy = tf.distribute.MultiWorkerMirroredStrategy()

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()

### Relevant log output
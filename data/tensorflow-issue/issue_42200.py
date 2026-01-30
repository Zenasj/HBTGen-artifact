from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from datetime import datetime
from packaging import version
import os
import tensorflow as tf
import numpy as np
import json
import tensorflow as  tf


os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["node72:12345", "node67:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '10,11')

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).cache().repeat().batch(batch_size)
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
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(multi_worker_dataset, epochs=20, steps_per_epoch=20,callbacks = [tboard_callback])

from datetime import datetime
from packaging import version
import os
import tensorflow as tf
import numpy as np
import json
import tensorflow as  tf


os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["node72:12345", "node67:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '10,11')

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # We need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).cache().repeat().batch(batch_size)
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
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(multi_worker_dataset, epochs=20, steps_per_epoch=20,callbacks = [tboard_callback])
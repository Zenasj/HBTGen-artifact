from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["server1:23531", "server2:41660"]
    },
    'task': {'type': 'worker', 'index': 0}
})

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["server1:23531", "server2:41660"]
    },
    'task': {'type': 'worker', 'index': 1}
})

#!/usr/bin/env python
# coding: utf-8

# # Multiple Worker Training
import tensorflow as tf
# Use tensorflow datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import os
import json

BUFFER_SIZE = 10000
BATCH_SIZE = 64

NUM_WORKERS = 2
# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

# Define TF_CONVIG environemnt variable
# This step create a environment variable that gives the location and ports available on the server to perform training.
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["server1:23531", "server2:41660"]
    },
    'task': {'type': 'worker', 'index': 1}
})

# This need to be called at the program startup
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name='mnist',with_info=True,as_supervised=True)

    #return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
    return datasets['train'].map(scale).shuffle(BUFFER_SIZE)


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

# #### Define a Strategy and distribute training
with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within 
    # `strategy.scope()`.
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
    multi_worker_model = build_and_compile_cnn_model()
    

multi_worker_model.fit(x=train_datasets, epochs=3)
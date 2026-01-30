from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

#!/usr/bin/env python

import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf
from slurm_utils import create_tf_config
tfds.disable_progress_bar()

BUFFER_SIZE = 10000
BATCH_SIZE = 64


def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)

    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,
                               3,
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


tfConfig = create_tf_config(gpus_per_task=2)
print('Used Config: {}'.format(tfConfig))
os.environ['TF_CONFIG'] = json.dumps(tfConfig)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print('Number of workers: {}\nParameter devices: {}\nWorkers: {}'.format(
    strategy.num_replicas_in_sync, strategy.extended.parameter_devices,
    strategy.extended.worker_devices))

# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(x=train_datasets, epochs=3)

for epoch in range(FLAGS.epoch):
        for train_data in train_dataset:
           ...
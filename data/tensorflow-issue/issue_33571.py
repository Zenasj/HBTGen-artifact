from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import os

tf.config.threading.set_inter_op_parallelism_threads(1)

datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True,
                           shuffle_files=False)

mnist_train, mnist_test = datasets['train'], datasets['test']
strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

num_train_examples = info.splits['train'].num_examples

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(
    num_train_examples).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

log_dir = "/tmp/tf_logs/fit/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")

with strategy.scope():
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

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                           histogram_freq=0,
                                           profile_batch=1)
    ]
    model.fit(train_dataset, epochs=3, steps_per_epoch=20, callbacks=callbacks)
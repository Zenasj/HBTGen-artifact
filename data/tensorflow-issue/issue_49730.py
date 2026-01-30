from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import random

from tensorflow import keras
import tensorflow as tf
import numpy as np

use_strategy = True
class Generator(tf.keras.utils.Sequence):
    def __init__(self):
        print("GENERATED")
        self.samples = np.random.rand(100000, 5, 20)
        self.labels = np.random.randint(2, size=(100000, 1))
        self.indices = list(range(0, 100000))
        random.Random().shuffle(self.indices)

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        return self.samples[self.indices[idx]], self.labels[self.indices[idx]]

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

if use_strategy:
    strategy = tf.distribute.MirroredStrategy()
    num_gpu = strategy.num_replicas_in_sync
else:
    num_gpu = 1
    
d = tf.data.Dataset.from_generator(Generator,
                                   output_signature=(tf.TensorSpec(shape=(5, 20)), tf.TensorSpec(shape=(1,))))
d = d.batch(32*num_gpu).prefetch(tf.data.AUTOTUNE).cache()

if use_strategy:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    d = d.with_options(options)

    with strategy.scope():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5, 20)))
        model.add(keras.layers.MaxPooling1D(pool_size=3, padding="same"))
        model.add(keras.layers.Conv1D(64, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5, 20)))
        model.add(keras.layers.MaxPooling1D(pool_size=3, padding="same"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation="relu"))
        model.add(keras.layers.Dense(1, activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(0.003),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
else:
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5, 20)))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding="same"))
    model.add(keras.layers.Conv1D(64, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5, 20)))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(1, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(0.003),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

model.fit(d, epochs=5)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from mpi_cluster_resolver import MPIClusterResolver

resolver = MPIClusterResolver()
communication = tf.distribute.experimental.CollectiveCommunication.NCCL
options = tf.distribute.experimental.CommunicationOptions(implementation=communication)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=options, cluster_resolver=resolver)

with strategy.scope():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

is_master = resolver.task_id == 0
verbose = 2 if is_master else 0
model.fit(train_data.repeat(), epochs=1, steps_per_epoch=10, verbose=verbose)
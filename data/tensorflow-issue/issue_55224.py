import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    batch_size = 3
    nb_samples = 10
    update_freq = 2
    inputs = tf.random.normal((nb_samples, 2))
    outputs = tf.random.normal((nb_samples, 10))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(batch_size)
    dataset_val = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(batch_size)
    nb_epochs = 5

    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    optimizer = tf.keras.optimizers.SGD()

    tensorboard_kwargs = dict(profile_batch=0, update_freq=update_freq, write_graph=False)

    tensorboard_callbacks = [
        tf.keras.callbacks.TensorBoard(tmpdir.join("tensorboard_1"), **tensorboard_kwargs),
        tf.keras.callbacks.TensorBoard(tmpdir.join("tensorboard_2"), **tensorboard_kwargs)
    ]
    model.compile(optimizer, "mse")
    model.fit(
        dataset,
        batch_size=batch_size,
        callbacks=tensorboard_callbacks,
        epochs=nb_epochs,
        validation_data=dataset_val
    )

import tensorflow as tf
def test(tmpdir):
    batch_size = 3
    nb_samples = 10
    update_freq = 2
    inputs = tf.random.normal((nb_samples, 2))
    outputs = tf.random.normal((nb_samples, 10))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(batch_size)
    dataset_val = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(batch_size)
    nb_epochs = 5

    writer = tf.summary.create_file_writer(str(tmpdir))

    with writer.as_default():
        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        optimizer = tf.keras.optimizers.SGD()

        tensorboard_kwargs = dict(profile_batch=0, update_freq=update_freq, write_graph=False)
        tensorboard_callbacks = [
            tf.keras.callbacks.TensorBoard(tmpdir.join("tensorboard_1"), **tensorboard_kwargs),
            tf.keras.callbacks.TensorBoard(tmpdir.join("tensorboard_2"), **tensorboard_kwargs)
        ]

    model.compile(optimizer, "mse")
    model.fit(
        dataset,
        batch_size=batch_size,
        callbacks=tensorboard_callbacks,
        epochs=nb_epochs,
        validation_data=dataset_val
    )
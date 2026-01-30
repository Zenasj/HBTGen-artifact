import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tempfile

import tensorflow as tf


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(session)


def main():
    batch_size = 3

    image_shape = (32, 32, 3)
    inputs = tf.random.uniform((batch_size, *image_shape))

    model = tf.keras.Sequential((
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='SAME',
            activation='linear'),
    ))

    _ = model(inputs)

    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
        tf.keras.models.save_model(model, fd.name, overwrite=True)
        model2 = tf.keras.models.load_model(fd.name, compile=False)

    print(model2.summary())


if __name__ == '__main__':
    main()

# Instead of the original
model2 = tf.keras.models.load_model(fd.name, compile=False)
# this works; the JSON is also stored in h5 file, but it cannot be easily accessed, so I take it directly from the original model
model2 = tf.keras.models.model_from_json(model.to_json())
model2(inputs)
model2.load_weights(fd.name)

import tempfile

import tensorflow as tf


def main():
    batch_size = 3

    image_shape = (32, 32, 3)
    inputs = tf.random.uniform((batch_size, *image_shape))

    model = tf.keras.Sequential((
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='SAME',
            activation='linear'),
    ))

    _ = model(inputs)

    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
        tf.keras.models.save_model(model, fd.name, overwrite=True)
        model2 = tf.keras.models.load_model(fd.name, compile=False)

    print(model2.summary())


if __name__ == '__main__':
    main()
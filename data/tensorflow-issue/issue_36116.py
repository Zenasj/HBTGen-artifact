import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import sys
import tensorflow as tf


@tf.function
def parse_entry(entry):
    feature_description = {
        "image_l": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "image_r": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "disparity_l": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32, allow_missing=True),
    }

    example = tf.io.parse_single_example(entry, feature_description)

    example["image_l"] = tf.io.decode_image(example["image_l"], channels=0, dtype=tf.dtypes.uint8)
    example["image_r"] = tf.io.decode_image(example["image_r"], channels=0, dtype=tf.dtypes.uint8)
    example["disparity_l"] = tf.reshape(example["disparity_l"], (540, 960, 1))

    return (example["image_l"], example["image_r"]), example["disparity_l"]


@tf.function
def normalise(x, y):
    img_l = tf.image.convert_image_dtype(x[0], dtype=tf.dtypes.float32, name="convert1")
    img_r = tf.image.convert_image_dtype(x[1], dtype=tf.dtypes.float32, name="convert2")

    return (img_l, img_r), y


@tf.function
def fixup_shape(x, y):
    x[0].set_shape([540, 960, 3])
    x[1].set_shape([540, 960, 3])
    y.set_shape([540, 960, 1])

    return x, y


@tf.function
def scale_output_resolution(x, y):
    if False:
        disparities = [
            tf.image.resize(
                y,
                size=(tf.math.divide(540, tf.math.pow(2, n)), tf.math.divide(960, tf.math.pow(2, n))),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            for n in range(1, 3)
        ]
    else:
        if False:
            disparities = [y for _ in range(1, 3)]
        else:
            disparities = y

    return x, disparities


def create_dataset(path):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_entry)
    dataset = dataset.map(normalise)
    dataset = dataset.map(fixup_shape)
    dataset = dataset.map(scale_output_resolution)
    dataset = dataset.batch(batch_size=1, drop_remainder=False)

    return dataset


class TestModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(TestModel, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(7, 7), strides=(2, 2), padding="same", name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), strides=(2, 2), padding="same", name="conv2")

    def call(self, inputs):
        x = self.conv1(tf.concat(inputs, axis=-1, name="concat"))
        y = self.conv2(x)

        return [x, y]


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)

    train_dataset = create_dataset(sys.argv[1])
    valid_dataset = create_dataset(sys.argv[2])

    model = TestModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.Accuracy],
    )
    model.fit(x=train_dataset, validation_data=valid_dataset, validation_steps=None, validation_freq=1, epochs=1)

import tensorflow as tf
import numpy as np

num_examples = 1000
input_shape = (10,)
loss_dim = 3
num_losses = 2

features = np.random.random(size=[num_examples] + list(input_shape))
labels = np.random.random(size=[num_examples, num_losses, loss_dim])

inputs = tf.keras.Input(shape=input_shape, name='features')
l1 = tf.keras.layers.Dense(loss_dim, activation='relu', name='l1')(inputs)
l2 = tf.keras.layers.Dense(loss_dim, activation='relu', name='l2')(l1)
model = tf.keras.Model(inputs=inputs, outputs=[l1, l2])

def loss(y, y_hat):
    return tf.abs(y - y_hat)

model.compile(optimizer='adam', loss=[loss, loss], loss_weights=[0.2, 1.])

model.fit(
    features, 
    labels,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

label_list = [l for l in labels.reshape(num_losses, num_examples, loss_dim)]
model.fit(
    features, 
    label_list,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)
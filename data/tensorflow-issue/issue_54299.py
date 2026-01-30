import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

batch_size = 1024

dataset, info = tfds.load(
    "imagenet2012:5.0.*",
    decoders={"image": tfds.decode.SkipDecoding()},
    split="train",
    with_info=True,
    data_dir="gs://my_data_bucket",
)
num_examples = info.splits["train"].num_examples


def _decode_and_center_crop(image_bytes):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height, image_width = shape[0], shape[1]
    image_size = 224

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + 32))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return tf.cast(image, dtype=tf.float32)


def preprocessing(data):
    return _decode_and_center_crop(data["image"]), data["label"]


dataset = (
    dataset.shuffle(num_examples // 2)
    .map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(1)
)

model = keras.models.Sequential(
    [
        keras.layers.GlobalMaxPool2D(input_shape=(224, 224, 3)),
        keras.layers.Dense(1000, activation="softmax"),
    ]
)

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")

model.fit(dataset, epochs=3)

for epoch in range(3):
    print(f"Epoch: {epoch}")
    data_iterator = iter(dataset)
    for batch in data_iterator:
        pass

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

batch_size = 1024

dataset, info = tfds.load(
    "imagenet2012:5.0.*",
    decoders={"image": tfds.decode.SkipDecoding()},
    split="train",
    with_info=True,
    data_dir="gs://my_data_bucket",
)
num_examples = info.splits["train"].num_examples


def _decode_and_center_crop(image_bytes):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height, image_width = shape[0], shape[1]
    image_size = 224

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + 32))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return tf.cast(image, dtype=tf.float32)


def preprocessing(data):
    return _decode_and_center_crop(data["image"]), data["label"]


dataset = (
    dataset.shuffle(num_examples // 2)
    .map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(1)
)

for epoch in range(3):
    print(f"Epoch: {epoch}")
    data_iterator = iter(dataset)
    for batch in data_iterator:
        pass

for epoch in range(3):
    for batch in iter(dataset):
        pass

for epoch in range(3):
    for batch in dataset:
        pass

import tensorflow as tf

# total size is 16gb: 1024*1024*256*4*16
# 4 comes from sizeof(float32)
example_size = 1024
batch_size = 1024 * 16
num_batches = 256

num_examples = batch_size * num_batches
shuffle_buffer_size = num_examples // 2

dataset = tf.data.Dataset.range(num_examples)

def make_data(_):
  return tf.random.uniform((example_size,), dtype=tf.float32)

dataset = dataset.map(make_data)

dataset = (
    dataset.shuffle(shuffle_buffer_size)
    .batch(batch_size)
    .prefetch(1)
)

for epoch in range(1000):
    print(f"Epoch: {epoch}")
    data_iterator = iter(dataset)
    for batch in data_iterator:
        pass
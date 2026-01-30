from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow_datasets as tfds


batch_size = 1024

dataset = tfds.load(
    "imagenet2012:5.0.0",
    decoders={"image": tfds.decode.SkipDecoding()},
    split="train",
    data_dir="gs://my-cloud-bucket",
)

val_dataset = tfds.load(
    "imagenet2012:5.0.0",
    decoders={"image": tfds.decode.SkipDecoding()},
    split="validation",
    data_dir="gs://my-cloud-bucket",
)


def _decode_and_center_crop(image_bytes):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]
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
    return tf.image.resize(image, [image_size, image_size], method="bicubic")


def preprocessing(data):
    return tf.cast(_decode_and_center_crop(data["image"]), tf.float32), data["label"]

dataset = (
    dataset.cache()
    .map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(1)
)

val_dataset = (
    val_dataset.cache()
    .map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(1)
)

with tf.distribute.MirroredStrategy().scope():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.GlobalMaxPool2D(input_shape=(224, 224, 3)),
            tf.keras.layers.Dense(1000, activation="softmax",),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
    )

model.fit(
    val_dataset, epochs=5, validation_data=dataset,
)
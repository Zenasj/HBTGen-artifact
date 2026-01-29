# tf.random.uniform((BATCH_SIZE, 299, 299, 3), dtype=tf.float32) ← inferred input shape from InceptionV3 input_shape=(299,299,3)

import os
import datetime
import pathlib
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32  # Assumed default batch size since it's referenced but not defined
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGESIZE = 299
epochs = 2

# Assuming datasetFilePath is the root directory containing class subfolders,
# this is needed by GetInput() to generate valid inputs matching model expectations.
datasetFilePath = "D:/TrainData/BalancedData"
datasetPath = pathlib.Path(datasetFilePath)
CLASS_NAMES = np.array([item.name for item in datasetPath.glob('*')])


def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == class_names


def parse_image(filename):
    # On Windows, the path uses backslash separator for raw filenames,
    # but tf.strings.split expects '/' typically, so handle with os.sep for portability.
    parts = tf.strings.split(filename, os.sep)
    label = get_label(filename, CLASS_NAMES)
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
    return image, label


def make_dataset_unbatched():
    list_ds = tf.data.Dataset.list_files(str(datasetPath / "*/*"))
    images_ds = list_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    images_ds = images_ds.shuffle(BATCH_SIZE)
    images_ds = images_ds.repeat(epochs)
    images_ds = images_ds.prefetch(buffer_size=AUTOTUNE)
    return images_ds


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the base InceptionV3 model, pretrained on ImageNet,
        # without the top layers, input shape 299x299x3
        self.base_model = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=(IMAGESIZE, IMAGESIZE, 3)
        )
        self.base_model.trainable = True

        # Classification head layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.predictions = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None):
        x = self.base_model(inputs, training=training)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.predictions(x)
        return x


def my_model_function():
    model = MyModel()
    base_learning_rate = 1e-5
    # Compile the model with Adam optimizer and categorical crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def GetInput():
    # Return a random input tensor simulating a batch of images
    # The batch size here can be BATCH_SIZE as per model training
    return tf.random.uniform(
        (BATCH_SIZE, IMAGESIZE, IMAGESIZE, 3), dtype=tf.float32
    )


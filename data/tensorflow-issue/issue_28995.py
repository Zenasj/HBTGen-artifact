from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_datasets as tfds

train, test = tfds.load(name="mnist", split=[tfds.Split.TRAIN, tfds.Split.TEST], as_supervised=True)

def scale(image, label):
    return tf.cast(image, tf.float32) / 255, label

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

model.fit(
    train.batch(256),
    validation_data=test.batch(256),
)
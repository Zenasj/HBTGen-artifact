from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compat.v1.train import Saver
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 64

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

shape = (28, 28, 1)
print(shape)

model = Sequential(
    [
        LSTM(128, input_shape=shape, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(lr=1e-3, decay=1e-5),
    metrics=["accuracy"],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

model.save("mnist_model.h5")

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compat.v1.train import Saver
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 64

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential()

model.add(LSTM(128, input_shape=x_train.shape[1:], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# model = Sequential(
#     [
#         LSTM(128, input_shape=x_train.shape[1:], return_sequences=True),
#         Dropout(0.2),
#         LSTM(128),
#         Dropout(0.2),
#         Dense(32, activation="relu"),
#         Dropout(0.2),
#         Dense(10, activation="softmax"),
#     ]
# )

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(lr=1e-3, decay=1e-5),
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    epochs=6,
    validation_data=(x_test, y_test),
)

model.save("mnist_model.h5")
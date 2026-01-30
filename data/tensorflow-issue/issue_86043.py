from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices(range(10)) \
    .map(lambda x: (
        tf.RaggedTensor.from_tensor(tf.zeros((x + 1, x + 1, 1))),
        0,
    )) \
    .batch(batch_size=4) \
    .prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=10, width=10),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Softmax(),
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy())

model.fit(ds)

import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)

ds = tf.data.Dataset.from_tensor_slices(range(10)) \
    .map(lambda x: (
        tf.RaggedTensor.from_tensor(tf.zeros((x + 1, x + 1, 1))),
        0,
    )) \
    .batch(batch_size=4) \
    .prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=10, width=10),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Softmax(),
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy())

model.fit(ds)

import keras
import tensorflow as tf

print(tf.__version__)
print(keras.__version__)

ds = tf.data.Dataset.from_tensor_slices(range(10)) \
    .map(lambda x: (
        tf.RaggedTensor.from_tensor(tf.zeros((x + 1, x + 1, 1))),
        0,
    )) \
    .batch(batch_size=4) \
    .prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.layers.Resizing(height=10, width=10),
    keras.layers.GlobalMaxPool2D(),
    keras.layers.Softmax(),
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy())

model.fit(ds)

import tf_keras
import tensorflow as tf

print(tf.__version__)
print(tf_keras.__version__)

ds = tf.data.Dataset.from_tensor_slices(range(10)) \
    .map(lambda x: (
        tf.RaggedTensor.from_tensor(tf.zeros((x + 1, x + 1, 1))),
        0,
    )) \
    .batch(batch_size=4) \
    .prefetch(tf.data.AUTOTUNE)

model = tf_keras.Sequential([
    tf_keras.layers.Resizing(height=10, width=10),
    tf_keras.layers.GlobalMaxPool2D(),
    tf_keras.layers.Softmax(),
])

model.compile(loss=tf_keras.losses.SparseCategoricalCrossentropy())

model.fit(ds)
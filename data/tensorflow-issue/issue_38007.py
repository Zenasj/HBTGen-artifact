import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
import numpy as np

num_samples = 100
height = 224
width = 224
num_classes = 50

strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0', '/GPU:1'])
with strategy.scope():
    parallel_model = Xception(weights=None,
                              input_shape=(height, width, 3),
                              classes=num_classes)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

### Works only for the first GPU of the
# parallel_model = Xception(weights=None,
#                           input_shape=(height, width, 3),
#                           classes=num_classes)
# parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

parallel_model.summary()
# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=16)

strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0'])

devices=['/GPU:1']

devices=['/GPU:0', '/GPU:1']

import tensorflow as tf

n = 12345
dtype = tf.float32
print(2 * n*n*32/8 / 1.e9)
with tf.device("/gpu:1"): # /gpu:0
    for i in range(100):
        matrix1 = tf.Variable(tf.random.uniform((n, n), dtype=dtype))
        matrix2 = tf.Variable(tf.random.uniform((n, n), dtype=dtype))
        product = tf.norm(tf.matmul(matrix1, matrix2))
        print(i, product)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
with tf.device("/gpu:1"):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
# tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices, 'GPU')

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model.compile(...)
    model.fit(...)
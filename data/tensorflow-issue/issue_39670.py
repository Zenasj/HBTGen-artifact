from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0", "/gpu:1"])

x = np.zeros((32, 28, 28))
y = np.zeros((32,))

with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

for i in range (2):
    model.train_on_batch(x, y)

print(tf.executing_eagerly())

import tensorflow as tf
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0", "/gpu:1"])

with strategy.scope():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(512)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),       
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.fit(
    train_dataset,
    epochs=2
)

print(tf.executing_eagerly())
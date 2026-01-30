import math
import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense)


@tf.function
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.math.square(y_pred - y_true))

if __name__ == "__main__":

    BATCH_SIZE_PER_SYNC = 4
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    global_batch_size = BATCH_SIZE_PER_SYNC * num_gpus
    print('num GPUs: {}, global batch size: {}'.format(num_gpus, global_batch_size))


    # fake data ------------------------------------------------------
    fakea = np.random.rand(global_batch_size, 10, 200, 200, 128).astype(np.float32)
    targets = np.random.rand(global_batch_size, 200, 200, 14).astype(np.float32)

    fakea = tf.constant(fakea)
    targets = tf.constant(targets)

    # tf.Dataset ------------------------------------------------------
    def gen():
        while True:
            yield (fakea, targets)

    dataset = tf.data.Dataset.from_generator(gen,
        (tf.float32, tf.float32),
        (tf.TensorShape(fakea.shape), tf.TensorShape(targets.shape)))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # training ------------------------------------------------------
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]
    training = True
    with strategy.scope():
        va = keras.Input(shape=(10, 200, 200, 128), dtype=tf.float32, name='va')
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(va)
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
        x = tf.reduce_max(x, axis=1, name='maxpool')                         
        b = Conv2D(14, kernel_size=3, padding='same')(x)
        model = keras.Model(inputs=va, outputs=b, name='net')
        optimizer = keras.optimizers.RMSprop()

        model.compile(optimizer=optimizer, loss=loss_fn)
        model.fit(x=dataset, epochs=10, steps_per_epoch=100, callbacks=callbacks)

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.SimpleRNN(10, return_sequences=True, input_shape=[None, 4]),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="nadam")

X_train = tf.random.uniform(shape=[100, 50, 4])
y_train = tf.random.uniform(shape=[100, 1])
model.fit(X_train, y_train)

for length in range(1, 20):
    X_new = tf.random.uniform([1, length, 4])
    model.predict(X_new)

from random import randint

import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv1D(8, 3))
model.build([None, 12, 1])

predict_tensors = [
    tf.random.normal([randint(1, 8), randint(4, 40), 1])
    for _ in range(10)
]
for t in predict_tensors:
    _ = model.predict(t)
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def normalize_manual(x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = x - mean
    var = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
    return x / (tf.sqrt(var + 1e-6))


def normalize_with_moments(x):
    mean, var = tf.nn.moments(x, [-1], keepdims=True)
    x = x - mean
    return x / tf.sqrt(var + 1e-6)


def run_model_custom(normalize=normalize_with_moments):
    inp = tf.keras.layers.Input(shape=(8,))
    x = inp
    x = tf.keras.layers.Dense(8)(x)
    x = normalize(x)
    out = tf.squeeze(tf.keras.layers.Dense(1)(x), axis=-1)
    model = tf.keras.Model(inputs=inp, outputs=out)

    x = tf.random.normal(shape=((100, 8)))
    y = tf.random.normal(shape=((100,)))
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(5)

    for x, y in dataset.take(1):
        with tf.GradientTape() as tape:
            out = model(x)
            tape.gradient(out, model.trainable_weights)
        break
    print('passed run_model_custom with {}'.format(normalize.__name__))


def train_model_fit(normalize=normalize_with_moments):
    inp = tf.keras.layers.Input(shape=(8,))
    x = inp
    x = tf.keras.layers.Dense(8)(x)
    x = normalize(x)
    out = tf.squeeze(tf.keras.layers.Dense(1)(x), axis=-1)
    model = tf.keras.Model(inputs=inp, outputs=out)

    x = tf.random.normal(shape=((100, 8)))
    y = tf.random.normal(shape=((100,)))
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(5)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')
    model.fit(dataset, epochs=1, steps_per_epoch=2)
    print('passed train_model_fit with {}'.format(normalize.__name__))


def compute_graph_tf(normalize=normalize_with_moments):
    x = tf.random.normal((10, 5))
    with tf.GradientTape() as tape:
        layer = tf.keras.layers.Dense(2)
        y = layer(x)
        y = tf.reduce_sum(normalize(y))
        tape.gradient(y, layer.trainable_weights)
    print('passed compute_graph_tf with {}'.format(normalize.__name__))


run_model_custom(normalize_with_moments)  # <----- fails

###########
# the below work
###########
train_model_fit(normalize_with_moments)
compute_graph_tf(normalize_with_moments)
run_model_custom(normalize_manual)
train_model_fit(normalize_manual)
compute_graph_tf(normalize_manual)
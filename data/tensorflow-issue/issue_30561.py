import random
from tensorflow import keras
from tensorflow.keras import layers

# coding: utf-8

"""Test script to measure runtime performances on mock data.

Set up the EAGER constant on line 18 to toggle the use of
Eager execution (provided it is not enabled by default,
otherwise change the instructions on lines 31 and 33).
Then, call `python3 <this_script.py>` to run it.
"""

import os
import time

import numpy as np
import tensorflow as tf


EAGER = False


def main(eager):
    """Set up and fit a model on mock data, with or without Eager execution.

    The model consists of a layer of 100 LSTM cells topped with
    a single dense unit with sigmoid activation. Its fitting is
    bound not to yield actual accuracy improvements, as the data
    used is purely random, but aims at measuring performances as
    to runtime.
    """
    if eager:
        tf.enable_v2_behavior()
    else:
        tf.enable_resource_variables()
    # Set up the classifier model using custom embedding units.
    inputs = tf.keras.Input((None, 100), dtype=tf.float32)
    lengths = tf.keras.Input((), dtype=tf.int64)
    mask = tf.sequence_mask(lengths)
    output = tf.keras.layers.LSTM(100)(inputs, mask=mask)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    model = tf.keras.Model([inputs, lengths], output)
    # Set up the training and validation data.
    dataset = setup_mock_dataset()
    train, valid = dataset.take(500), dataset.skip(500)
    # Fit the model.
    model.compile('adam', 'binary_crossentropy', ['accuracy'])
    history = model.fit(
        train.repeat(), steps_per_epoch=500, epochs=10,
        validation_data=valid.repeat(), validation_steps=100
    )


def setup_mock_dataset(n_batches=600, batch_size=32):
    """Return a tf.data.Dataset yielding batches of random data.

    The input data consists of a couple of tensors, one with
    zero-padded sequences of vectors of size 100, the other
    with the true sequence lengths. The target data consists
    of sequence-wise binary values.
    """
    # Generate some random (mock) input and target data.
    n_samples = n_batches * batch_size
    lengths = 1 + np.random.choice(100, size=n_samples, replace=True)
    inputs = np.random.normal(size=(lengths.sum(), 100))
    targets = np.random.choice(2, size=(n_samples, 1), replace=True)
    # Set up a generator yielding shuffled training samples.
    def generator():
        """Yield individual training samples."""
        nonlocal inputs, targets, lengths, n_samples
        start = 0
        for i in range(n_samples):
            end = start + lengths[i]
            yield (inputs[start:end], lengths[i]), targets[i]
            start = end
    # Set up a tensorflow Dataset based on the previous.
    output_shapes = (
        (tf.TensorShape((None, 100)), tf.TensorShape(())), tf.TensorShape(1)
    )
    dataset = tf.data.Dataset.from_generator(
        generator, ((tf.float32, tf.int64), tf.int64), output_shapes
    )
    # Have the dataset output data batches, and return it.
    return dataset.padded_batch(batch_size, output_shapes)


if __name__ == '__main__':
    start = time.clock()
    main(EAGER)
    duration = time.clock() - start
    min, sec = duration // 60, duration % 60
    print('TOTAL DURATION: %im%i' % (min, sec))

if not eager:
    tf.compat.v1.disable_eager_execution()
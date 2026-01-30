import math
import random
from tensorflow import keras
from tensorflow.keras import layers

# coding: utf-8

"""Minimal example script for an RNN issue within custom train loops."""

import numpy as np
import tensorflow as tf


class StackedModel(tf.keras.Model):
    """Minimal gradient stacking example Model subclass."""

    def train_step(self, data):
        # NOTE: here we assume data is a single (x, y) tuple
        #       in order to provide with a minimal example
        inputs, y_true = data
        size = tf.shape(inputs)[0] // 4
        # Compute gradients on the batch's first quarter.
        gradients = self._get_gradients(inputs[:size], y_true[:size])
        # Define a process to compute and stack gradients.
        def process_quarter(idx, gradients):
            """Compute gradients on a data sub-batch and stack them."""
            grads_loc = self._get_gradients(
                inputs[idx * size:(idx + 1) * size],
                y_true[idx * size:(idx + 1) * size]
            )
            gradients = [
                self._add_gradients(a, b) for a, b in zip(gradients, grads_loc)
            ]
            return tf.add(idx, 1), gradients
        # Iteratively process the remaining data quarters using the former.
        _, gradients = tf.while_loop(
            cond=lambda idx, _: tf.math.less(idx, 4),
            body=process_quarter,
            loop_vars=[tf.constant(1), gradients],
            parallel_iterations=1
        )
        # Apply the aggregated gradients.
        grads_and_vars = zip(gradients, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)
        # Return the current values of the loss and metrics.
        return {m.name: m.result() for m in self.metrics}

    def _get_gradients(self, inputs, y_true):
        """Compute gradients for given (x, y) data."""
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compiled_loss(y_true, y_pred)
        return tape.gradient(loss, self.trainable_variables)

    @staticmethod
    def _add_gradients(grad_a, grad_b):
        """Return the sum of two gradient objects (Tensor of IndexedSlices)."""
        if not isinstance(grad_b, type(grad_a)):
            raise TypeError("Trying to add objects of distinct types.")
        if isinstance(grad_a, tf.Tensor):
            return tf.add(grad_a, grad_b)
        if isinstance(grad_a, tf.IndexedSlices):
            values = tf.concat([grad_a.values, grad_b.values], axis=0)
            indices = tf.concat([grad_a.indices, grad_b.indices], axis=0)
            return tf.IndexedSlices(values, indices, grad_a.dense_shape)


def build_example_model(run_eagerly, avoid_mask):
    """Return a keras Model for binary classification of tokens sequences.

    This model expects an input batch of tokens, with zero values
    being treated as padding, and thus masked. An embedding layer
    encodes the tokens into vectors in R^{128}, then a LSTM layer
    produces sequence-wise vectors in R^{128}, which are finally
    transformed into binary probabilities by a dense layer.
    """
    inputs = tf.keras.Input((None,), dtype=tf.int32)
    emb = tf.keras.layers.Embedding(
        input_dim=200,
        output_dim=128,
        mask_zero=True
    )
    rnn = tf.keras.layers.LSTM(128)
    out = tf.keras.layers.Dense(2, 'softmax')
    embedding = emb(inputs)
    if avoid_mask:
        embedding = rnn(embedding, mask=None)
    else:
        embedding = rnn(embedding)  # mask is passed implicitly
    model = StackedModel(inputs, out(embedding))
    model.compile(loss='binary_crossentropy', run_eagerly=run_eagerly)
    return model


def build_example_dataset():
    """Return a tf.data.Dataset of batched right-padded tokens sequences."""
    # Define a random tokens sequences generator.
    def generator():
        """Yield sequences of 8 to 32 random ints in (1, 200(, plus a label."""
        sizes = 8 + np.random.choice(24, size=640, replace=True)
        for i in range(640):
            seq = 1 + np.random.choice(199, size=sizes[i], replace=True)
            lab = tf.one_hot(np.random.choice(2), depth=2)
            yield (seq, lab)
    # Set up and return a Dataset made of batches of 32 padded sequences.
    dst = tf.data.Dataset.from_generator(
        generator,
        output_shapes=((None,), (2,)),
        output_types=(tf.int32, tf.float32)
    )
    return dst.padded_batch(32, padded_shapes=((None,), (2,)))


def main():
    """Minimal demonstration script."""
    dst = build_example_dataset().repeat()
    print('Running eagerly without masking at LSTM.')
    model = build_example_model(run_eagerly=True, avoid_mask=True)
    model.fit(dst, steps_per_epoch=20, epochs=3)
    print('Running eagerly with masking at LSTM.')
    model = build_example_model(run_eagerly=True, avoid_mask=False)
    model.fit(dst, steps_per_epoch=20, epochs=3)
    print('Running in graph mode without masking at LSTM.')
    model = build_example_model(run_eagerly=False, avoid_mask=True)
    model.fit(dst, steps_per_epoch=20, epochs=3)
    print('Running in graph mode with masking at LSTM -- prepare for failure.')
    model = build_example_model(run_eagerly=False, avoid_mask=False)
    model.fit(dst, steps_per_epoch=20, epochs=3)


if __name__ == '__main__':
    main()
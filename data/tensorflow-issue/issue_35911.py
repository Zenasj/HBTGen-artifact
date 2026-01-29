# tf.random.uniform((16, 1), dtype=tf.float32) ← input shape inferred from __getitem__ returning (16, 1) batches

import tensorflow as tf
import numpy as np


class MyModel(tf.keras.Model):
    """
    A simple model to demonstrate the usage of a custom tf.keras.utils.Sequence
    and reproduce the issue described where Sequence.on_epoch_end() is not
    properly called by model.fit in certain TF versions.

    This model is a single Dense layer with softmax activation.
    """

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, activation='softmax', input_shape=(1,))

    def call(self, inputs, training=False):
        return self.dense(inputs)


def my_model_function():
    """
    Returns an instance of MyModel with compilation as per the original example.
    """
    model = MyModel()
    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    """
    Returns a random input tensor that matches input expected by MyModel:
    shape (batch_size=16, input_dim=1), dtype float32.
    """
    return tf.random.uniform(shape=(16, 1), dtype=tf.float32)


class ZerosFirstEpochOnesAfter(tf.keras.utils.Sequence):
    """
    A Sequence yielding batches of zeros for the first epoch,
    and ones for subsequent epochs. Demonstrates the on_epoch_end callback
    which is supposed to be called after each epoch to update internal state.

    This class reproduces the issue that on_epoch_end is not called automatically
    when used with model.fit in TensorFlow 2.1.0, requiring a callback workaround.
    """

    def __init__(self):
        self.is_epoch_0 = True
        self.batch_size = 16

    def __len__(self):
        # We simulate 2 batches per epoch
        return 2

    def on_epoch_end(self):
        # This should be called automatically by model.fit at epoch end,
        # but in older TF versions, it is not.
        print('on_epoch_end')
        self.is_epoch_0 = False

    def __getitem__(self, idx):
        if self.is_epoch_0:
            print("First epoch")
            x = np.zeros((self.batch_size, 1), dtype=np.float32)
            y = np.zeros((self.batch_size,), dtype=np.float32)
        else:
            x = np.ones((self.batch_size, 1), dtype=np.float32)
            y = np.ones((self.batch_size,), dtype=np.float32)
        return x, y


class OnEpochEndCallback(tf.keras.callbacks.Callback):
    """
    Workaround callback to force calling the `on_epoch_end` method of a
    tf.keras.utils.Sequence, since in some TensorFlow versions (e.g., 2.1.0)
    model.fit does NOT call Sequence.on_epoch_end automatically.

    Usage:
        sequence = ZerosFirstEpochOnesAfter()
        model.fit(sequence, epochs=5, callbacks=[OnEpochEndCallback(sequence)])
    """

    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def on_epoch_end(self, epoch, logs=None):
        # Explicitly call the Sequence's on_epoch_end method
        self.sequence.on_epoch_end()


# Notes on usage (not included in code execution):
#
# This code reconstructs the issue described in the GitHub issue #35911,
# where the Sequence.on_epoch_end callback is not called automatically by
# model.fit, causing iteration state not to update as expected.
#
# The ZerosFirstEpochOnesAfter sequence yields zeros for the first epoch
# and ones afterward, changing based on an internal flag toggled by on_epoch_end.
#
# In TensorFlow 2.1.0, the workaround is to register the sequence's on_epoch_end
# call explicitly via a Callback like OnEpochEndCallback.
#
# The model and input shape are reconstructed faithfully from the minimal provided
# example. This code is compatible with TF 2.20.0 and can be jit-compiled without issue.


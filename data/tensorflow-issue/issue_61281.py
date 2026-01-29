# tf.random.uniform((B,), dtype=tf.float32) ← Based on the issue, inputs are 1D float tensors of arbitrary length (potentially empty)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable layers needed, this model mimics MeanAbsoluteError metric with fix for empty tensors

    @tf.function
    def call(self, inputs):
        """
        inputs: tuple of (y_true, y_pred)
          both are 1D float tensors, possibly empty
        Returns:
          The mean absolute error as scalar tensor (float32).
          If input tensors are empty, returns 0.0 (instead of NaN).
        """

        y_true, y_pred = inputs

        # Compute number of elements
        n = tf.size(y_true)

        # If empty, return 0.0 directly to avoid NaN
        def empty_case():
            return tf.constant(0.0, dtype=tf.float32)

        # Else compute MAE normally
        def non_empty_case():
            return tf.reduce_mean(tf.abs(y_true - y_pred))

        return tf.cond(n > 0, non_empty_case, empty_case)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tuple (y_true, y_pred) of 1D float tensors
    # Random size between 0 and 10 elements to potentially test empty case
    size = tf.random.uniform(shape=(), minval=0, maxval=11, dtype=tf.int32)
    y_true = tf.random.uniform((size,), minval=0, maxval=100, dtype=tf.float32)
    y_pred = tf.random.uniform((size,), minval=0, maxval=100, dtype=tf.float32)
    return (y_true, y_pred)


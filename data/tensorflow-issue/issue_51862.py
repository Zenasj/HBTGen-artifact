# tf.random.uniform((B, ...), dtype=tf.float32) ← Input shape is not explicitly defined in the issue, so we assume a generic batch of predictions/labels shape

import tensorflow as tf
from tensorflow import keras
import math

class MyModel(tf.keras.Model):
    """
    This model encapsulates two approaches to custom metrics for demonstration:
    1) A classical subclassed Metric for PSNR using tf.Variable with `assign`
    2) A functional metric approach for PSNR that directly computes the metric without variable accumulation

    The forward pass compares these two PSNR calculations for a given (y_true, y_pred) pair,
    returning whether they match within a numerical tolerance.

    This is to illustrate the discussion in the issue on different ways to create custom metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Using a tf.Variable to accumulate PSNR values
        self.psnr_var = self.add_weight(name='psnr', initializer='zeros', trainable=False)

    def update_psnr_variable(self, y_true, y_pred):
        """
        Update the internal PSNR variable using assign, not assign_add,
        reflecting the notion that assign may be more appropriate than assign_add
        for metrics that are not sums.
        """
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        mse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_f, y_pred_f))
        # We avoid divide by zero by adding a small epsilon
        epsilon = 1e-8
        mse_safe = tf.maximum(mse, epsilon)
        psnr_val = 10.0 * tf.math.log( (10000.0**2) / mse_safe ) / tf.math.log(10.0)
        # Instead of self.psnr_var.assign_add(psnr_val), we use assign to show different semantics
        self.psnr_var.assign(psnr_val)

    def functional_psnr(self, y_true, y_pred):
        """
        A functional-style PSNR metric just returning the metric, no variable state.
        This illustrates the "easy" way to implement metrics as a function.
        """
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        mse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true_f, y_pred_f))
        epsilon = 1e-8
        mse_safe = tf.maximum(mse, epsilon)
        psnr_val = 10.0 * tf.math.log( (10000.0**2) / mse_safe ) / tf.math.log(10.0)
        return psnr_val

    def call(self, inputs, training=None):
        """
        Here inputs is expected to be a tuple (y_true, y_pred):

        - We update the internal variable metric with assign during call.
        - We compute the functional metric.
        - Then we compare both PSNR values with a small tolerance and return a boolean scalar
          tensor indicating whether they are close (within 1e-5).

        This fusion illustrates usage of both approaches side-by-side and a comparison as per the issue discussion.
        """
        y_true, y_pred = inputs

        # Update the variable metric state (simulate update_state call)
        self.update_psnr_variable(y_true, y_pred)

        # Compute functional metric on the fly
        psnr_func = self.functional_psnr(y_true, y_pred)

        # Compare internal variable metric and functional metric with tolerance
        is_close = tf.math.abs(self.psnr_var - psnr_func) < 1e-5

        return is_close

def my_model_function():
    """
    Return a new instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Return a tuple (y_true, y_pred) of random tensors matching typical shapes for PSNR metric usage.

    Assumptions:
    - Batch size of 4
    - Images of size 64 x 64
    - Single channel (grayscale) to simplify PSNR calculation

    The values are floats in [0, 255] range (common image intensity range for PSNR).
    """
    batch_size = 4
    height = 64
    width = 64
    channels = 1

    y_true = tf.random.uniform(shape=(batch_size, height, width, channels), 
                               minval=0.0, maxval=255.0, dtype=tf.float32)
    y_pred = tf.random.uniform(shape=(batch_size, height, width, channels), 
                               minval=0.0, maxval=255.0, dtype=tf.float32)

    return (y_true, y_pred)


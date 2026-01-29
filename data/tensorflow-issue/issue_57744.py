# tf.constant(shape=[1], dtype=tf.float32) ← input is a 1-D scalar tensor

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters or layers needed; this module just performs the computation
        
    @tf.function(jit_compile=True)
    def call(self, x):
        # Mimic the original MyModule behavior:
        # output = log(x^x)
        # The issue: under XLA, log(pow(x, x)) is simplified incorrectly as x * log(abs(x))
        # which yields numeric inaccuracies (wrong output for negative x with fractional powers).
        #
        # The naive mathematically correct behavior is:
        #   output = log(pow(x, x)) = x * log(x) where domain issues produce nan if invalid
        # But due to algebraic simplification in XLA, we get:
        #   output = x * log(abs(x))
        #
        # We'll implement the direct computation without abs to replicate correct numerics
        # as a workaround / demonstration.
        #
        # To avoid runtime domain errors, use tf.where to handle x <= 0 safely:
        # - For x > 0, output = x * log(x)
        # - For x <= 0, output = nan (to mimic math domain errors)

        # Use float32 computation as in the original examples
        x = tf.cast(x, tf.float32)
        # Compute log(x) where valid; otherwise nan
        log_x = tf.math.log(x)
        nan_tensor = tf.fill(tf.shape(x), tf.constant(float('nan'), dtype=tf.float32))
        safe_log_x = tf.where(x > 0, log_x, nan_tensor)
        out = x * safe_log_x
        return out


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()


def GetInput():
    # Return a tensor matching model input:
    # The original input shape was [1] with scalar float value -1.5 to demonstrate domain error:
    # We'll replicate that exact example with shape=[1], dtype=float32
    return tf.constant([-1.5], shape=[1], dtype=tf.float32)


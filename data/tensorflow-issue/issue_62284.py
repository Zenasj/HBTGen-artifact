# tf.random.normal((8, 8, 8), dtype=tf.float32) ← input shape inferred from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We will encapsulate two modes of operation:
        # 1) AdjustContrastV2 call without jit_compile (eager)
        # 2) AdjustContrastV2 call with jit_compile=True
        # This reflects the issue: the same input gives different outputs due to 
        # the use of a random contrast_factor inside the tf.function with jit_compile.
        # To fairly compare, we will fix contrast_factor to a constant (e.g., 0.2)
        # because the root cause was random contrast_factor differing across executions.
        #
        # For demonstration, model will output boolean tensor indicating if outputs 
        # from both methods are close within atol and rtol.

        # Contrast factor fixed to 0.2 as per analysis in the issue comments
        self.contrast_factor = tf.constant(0.2, dtype=tf.float32)

    def call(self, x):
        # Compute AdjustContrastV2 in 'eager' style (simulate no jit)
        # We call it directly with fixed contrast_factor

        # Eager (no jit) version:
        def no_jit_adjust_contrast(images):
            return tf.raw_ops.AdjustContrastv2(images=images, contrast_factor=self.contrast_factor)

        # JIT compiled version:
        @tf.function(jit_compile=True)
        def jit_adjust_contrast(images):
            return tf.raw_ops.AdjustContrastv2(images=images, contrast_factor=self.contrast_factor)

        no_jit_out = no_jit_adjust_contrast(x)
        jit_out = jit_adjust_contrast(x)

        # Compute elementwise closeness
        # Use tf.debugging.assert_near equivalent logic manually:
        # output boolean tensor: True if close, False otherwise
        atol = 1e-3
        rtol = 1e-3
        difference = tf.abs(tf.cast(no_jit_out, tf.float64) - tf.cast(jit_out, tf.float64))
        tolerance = atol + rtol * tf.abs(tf.cast(jit_out, tf.float64))
        comparison = tf.less_equal(difference, tolerance)

        # Also return the two outputs for reference
        # Output a dictionary-like structure as tuple: (comparison_bool_tensor, no_jit_out, jit_out)
        # For architectural clarity, pack outputs in a dictionary via tf.nest
        return comparison, no_jit_out, jit_out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (8, 8, 8) float32 as per the issue example
    return tf.random.normal([8, 8, 8], dtype=tf.float32)


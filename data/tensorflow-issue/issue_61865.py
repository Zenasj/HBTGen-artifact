# tf.gather input shape inferred as (256,) from example, indices shape (6,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model demonstrates tf.gather behavior under jit_compile=True and False,
    specifically highlighting the difference for out-of-range indices handling.

    The original issue:
    When using tf.gather with out-of-range indices, the results differ:
      - With jit_compile=True (XLA), out-of-range indices are clamped/return 0 by XLA semantics.
      - Without jit_compile (normal TF eager or graph), GPU returns 0, CPU raises error (or behaves differently).
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # The indices used in the original snippet including out of range indices
        self.indices = tf.constant([5, 8, 7, 16, 256, 123], dtype=tf.int32)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs shape: (256,)
        # tf.gather with out-of-range indices under JIT/XLA compiles clamps indices within bounds (spec)
        gathered = tf.gather(inputs, indices=self.indices, axis=0)
        return gathered

def my_model_function():
    return MyModel()

def GetInput():
    # Input is a vector of length 256 of ones (float32), matching original example
    return tf.ones((256,), dtype=tf.float32)


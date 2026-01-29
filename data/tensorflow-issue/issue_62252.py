# tf.random.uniform((10, 9, 8), dtype=tf.bfloat16) ← inferred input shape and dtype from issue repro

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply Relu6 then Sigmoid using raw ops as in the issue reproducible code.
        x_jit = tf.raw_ops.Relu6(features=x)
        x_jit = tf.raw_ops.Sigmoid(x=x_jit)
        return x_jit

    @tf.function(jit_compile=False)
    def call_nojit(self, x):
        # Same operations without JIT compilation for comparison.
        x_nojit = tf.raw_ops.Relu6(features=x)
        x_nojit = tf.raw_ops.Sigmoid(x=x_nojit)
        return x_nojit

    @tf.function(jit_compile=True)
    def call_compare(self, x):
        # Provide a method to compare jit_compile=True vs False results numerically
        nojit = self.call_nojit(x)
        jit = self.call(x)

        # Cast to float64 for precision comparison as in the issue
        nojit64 = tf.cast(nojit, tf.float64)
        jit64 = tf.cast(jit, tf.float64)

        # Compute max absolute and relative difference
        abs_diff = tf.reduce_max(tf.abs(nojit64 - jit64))
        rel_diff = tf.reduce_max(tf.abs((nojit64 - jit64) / tf.maximum(tf.abs(nojit64), 1e-12)))

        # Return differences and boolean if within tolerance (from issue tolerances)
        atol = 0.001
        rtol = 0.001
        within_tol = tf.logical_and(abs_diff <= atol, rel_diff <= rtol)
        return within_tol, abs_diff, rel_diff


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tf.Tensor with shape [10, 9, 8] and dtype bfloat16,
    # matching the issue reproducible example input.
    # Use random normal distribution as in issue example.
    return tf.random.normal([10, 9, 8], dtype=tf.bfloat16)


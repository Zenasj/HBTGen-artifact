# tf.random.normal((10, 9, 8), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply tf.raw_ops.Ndtri, which is the inverse of the normal CDF (probit function)
        # This corresponds to the original Network __call__ with JIT compilation enabled.
        return tf.raw_ops.Ndtri(x=x)

    @tf.function
    def no_jit_call(self, x):
        # Same operation without jit_compile for comparison
        return tf.raw_ops.Ndtri(x=x)

    def call_compare(self, x):
        # Run both JIT and no-JIT versions and compare results
        y_jit = self.call(x)
        y_nojit = self.no_jit_call(x)

        # Cast to float64 for numerical stability in comparison
        y_jit_d = tf.cast(y_jit, tf.float64)
        y_nojit_d = tf.cast(y_nojit, tf.float64)

        # Compute absolute difference and check if they are close within tolerance
        atol = 0.001
        rtol = 0.001
        close = tf.math.less_equal(tf.abs(y_jit_d - y_nojit_d), atol + rtol * tf.abs(y_nojit_d))

        # Return a dictionary with result tensors and overall "all close" boolean
        return {
            'jit_result': y_jit,
            'nojit_result': y_nojit,
            'all_close': tf.reduce_all(close),
            'difference': y_jit_d - y_nojit_d,
            'close_mask': close,
        }

def my_model_function():
    return MyModel()

def GetInput():
    # According to the issue, input shape is [10, 9, 8], dtype float32
    # Input must be between 0 and 1 for Ndtri (inverse normal CDF; input is probability)
    # Use uniform random values in open interval (0,1) to avoid inf/-inf or NaN from Ndtri
    return tf.random.uniform(shape=(10, 9, 8), minval=1e-6, maxval=1-1e-6, dtype=tf.float32)


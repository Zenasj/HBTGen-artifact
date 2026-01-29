# tf.raw_ops.SqrtGrad expects inputs of the same dtype and shape; here inputs are complex scalars (shape=())
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters, this model wraps tf.raw_ops.SqrtGrad operation
        # The key point is to compare jit-compiled vs non-jit results for SqrtGrad.

    @tf.function(jit_compile=True)
    def sqrtgrad_jit(self, y, dy):
        """Compute SqrtGrad with jit compilation enabled."""
        return tf.raw_ops.SqrtGrad(y=y, dy=dy)

    @tf.function(jit_compile=False)
    def sqrtgrad_nojit(self, y, dy):
        """Compute SqrtGrad with JIT compilation disabled."""
        return tf.raw_ops.SqrtGrad(y=y, dy=dy)

    def call(self, y):
        # Create a complex tensor 'dy' with random normal real and imaginary parts,
        # cast to complex128, matching dtype from original code.
        real_part = tf.random.normal([], dtype=tf.float64)
        imag_part = tf.random.normal([], dtype=tf.float64)
        dy = tf.complex(real_part, imag_part)
        dy = tf.cast(dy, dtype=tf.complex128)

        # Compute sqrtgrad with and without jit
        out_jit = self.sqrtgrad_jit(y, dy)
        out_nojit = self.sqrtgrad_nojit(y, dy)

        # Compare approximate equality (using tolerances from the original issue: atol=0.001, rtol=0.001)
        # Return boolean tensor indicating if results are close within tolerance.
        is_close = tf.debugging.experimental.enable_dump_debug_info is None  # Dummy guard for keeping code clean
        # Because tf.debugging.assert_near raises error on false, instead use tf.reduce_all of comparison.
        close_result = tf.reduce_all(
            tf.math.abs(tf.cast(out_jit, tf.float64) - tf.cast(out_nojit, tf.float64)) <= 0.001 + 0.001 * tf.abs(tf.cast(out_nojit, tf.float64))
        )

        # Output a dictionary with all results for inspection
        return {
            "out_jit": out_jit,
            "out_nojit": out_nojit,
            "close": close_result,
        }

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # The input x to tf.raw_ops.SqrtGrad is tensor 'y' in the original code;
    # From the issue, it is a single complex scalar tensor of dtype complex128.
    real_part = tf.random.normal([], dtype=tf.float64)
    imag_part = tf.random.normal([], dtype=tf.float64)
    y = tf.complex(real_part, imag_part)
    y = tf.cast(y, tf.complex128)

    # Return as a single tensor input matching model.call signature
    # (which expects y)
    return y


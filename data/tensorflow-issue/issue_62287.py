# tf.random.normal((10, 9, 8), dtype=tf.bfloat16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model internally implements two branches:
        # one uses jit_compile=True to compute Cos -> Erfc,
        # the other computes without jit compilation.
        # The outputs are compared for near-equality with given atol and rtol.
        # This fusion encapsulates both behaviors described in the issue.

    @tf.function(jit_compile=True)
    def _with_jit(self, x):
        # Compute Cos then Erfc with JIT compilation enabled
        x = tf.raw_ops.Cos(x=x)
        x = tf.raw_ops.Erfc(x=x)
        return x

    @tf.function
    def _without_jit(self, x):
        # Compute Cos then Erfc without JIT compilation
        x = tf.raw_ops.Cos(x=x)
        x = tf.raw_ops.Erfc(x=x)
        return x

    def call(self, inputs):
        # Inputs is expected to be a tensor of shape (10, 9, 8), dtype bfloat16
        # Run both JIT and non-JIT computations
        out_jit = self._with_jit(inputs)
        out_nojit = self._without_jit(inputs)

        # Cast outputs to float64 for stable comparison
        out_jit_f64 = tf.cast(out_jit, tf.float64)
        out_nojit_f64 = tf.cast(out_nojit, tf.float64)

        # Use tf.debugging.assert_near for validation would raise error on mismatch
        # Here, instead of raising, produce a boolean tensor indicating closeness
        is_close = tf.reduce_all(
            tf.math.abs(out_jit_f64 - out_nojit_f64) <= 0.001 + 0.001 * tf.math.abs(out_nojit_f64)
        )
        # Output a dict containing both outputs and the comparison flag
        return {
            "output_jit": out_jit,
            "output_no_jit": out_nojit,
            "are_outputs_close": is_close,
            "diff_abs": tf.abs(out_jit_f64 - out_nojit_f64),
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return input matching the model's expected input signature
    # Shape: (10, 9, 8), dtype: bfloat16, values sampled from normal distribution
    return tf.random.normal(shape=(10, 9, 8), dtype=tf.bfloat16)


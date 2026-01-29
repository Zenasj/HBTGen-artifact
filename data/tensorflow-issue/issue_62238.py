# tf.random.uniform((), dtype=tf.complex128)  ← Input is a scalar complex128 tensor (single value)

import tensorflow as tf

def replace_special_values(tensor):
    # Replace NaNs with zeros
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

    # Replace positive infinities with large number (100)
    tensor = tf.where(tf.math.is_inf(tensor) & tf.math.greater_equal(tensor, 0), tf.constant(100, dtype=tensor.dtype), tensor)

    # Replace negative infinities with small number (-100)
    tensor = tf.where(tf.math.is_inf(tensor) & tf.math.less(tensor, 0), tf.constant(-100, dtype=tensor.dtype), tensor)
    return tensor


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters, just raw_ops usage

    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply tf.raw_ops.Sin then tf.raw_ops.Asinh as per the issue reproduction
        # Supports complex64, complex128, float types as per source notes
        x_sin = tf.raw_ops.Sin(x=x)
        x_asinh = tf.raw_ops.Asinh(x=x_sin)
        return x_asinh

def my_model_function():
    # Return an instance of MyModel for usage
    return MyModel()

def GetInput():
    # Return a scalar complex128 tensor as input, matching original example
    # The original example used the complex number: (900108.2563231019-417958.5363911558j)
    val = tf.constant(900108.2563231019 - 417958.5363911558j, dtype=tf.complex128)
    return val


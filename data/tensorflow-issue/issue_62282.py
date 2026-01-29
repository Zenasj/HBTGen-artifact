# tf.random.normal((10, 9, 8), dtype=tf.bfloat16)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights in this model, purely applies ops

    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply the two ops in sequence with jit_compile enabled
        x_jit = tf.raw_ops.Cos(x=x)
        x_jit = tf.raw_ops.Selu(features=x_jit)
        
        # Apply the same ops without jit_compile (simulate by running eagerly inside the graph)
        # TensorFlow does not provide direct ops to disable jit inside a tf.function,
        # so we approximate by running on CPU outside jit context (non-jit).
        # For illustration, we just reuse raw ops here as no direct switch available.
        # Instead, we simulate by detaching and running again without jit_compile using tf.numpy_function
        def no_jit_ops(x_np):
            # numpy implementation for comparison
            import numpy as np
            x_np = np.cos(x_np)
            # selu implementation as per TF Selu:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            x_np = np.where(x_np > 0, scale * x_np, scale * alpha * (np.exp(x_np) - 1))
            return x_np

        x_no_jit = tf.numpy_function(no_jit_ops, [tf.cast(x, tf.float32)], tf.float32)
        x_no_jit = tf.cast(x_no_jit, x.dtype)

        # Compare the two outputs within tolerances and output a boolean tensor: True if close
        close = tf.reduce_all(
            tf.math.abs(tf.cast(x_jit, tf.float64) - tf.cast(x_no_jit, tf.float64)) 
            <= 0.001 + 0.001 * tf.math.abs(tf.cast(x_no_jit, tf.float64))
        )
        # Returning the comparison result (bool scalar) and both outputs for inspection
        # To comply with TF2 model call returning tensor, we return a tuple (or dictionary)
        # Though Keras Models typically expect tensor output, returning a dict is supported,
        # or form output as concatenated tensor with specific meaning
        # Here we return a dict for clarity:
        return {
            "jit_output": x_jit,
            "no_jit_output": x_no_jit,
            "are_close": close,
        }

def my_model_function():
    return MyModel()

def GetInput():
    # Produce input tensor matching the input from the issue:
    # shape=[10,9,8], dtype=tf.bfloat16, normal distribution
    return tf.random.normal((10, 9, 8), dtype=tf.bfloat16)


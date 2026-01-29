# tf.random.normal((9, 8), dtype=tf.float32) → used to generate inputs for tf.raw_ops.ReciprocalGrad 'dy' and 'y' tensors

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model will encapsulate two call modes of tf.raw_ops.ReciprocalGrad:
        # 1) Without JIT compilation (eager or normal tf.function call)
        # 2) With JIT compilation (jit_compile=True)
        # Then it will compare their outputs and return the difference.
        # Inputs to the model: a tensor used as dy for ReciprocalGrad.
        # Internally, a fixed random complex tensor "y" is generated each call.
    
    @tf.function
    def call_without_jit(self, x):
        # Create a complex64 tensor as y, same shape as x, freshly on each call
        real_part = tf.random.normal([9, 8], dtype=tf.float32)
        imag_part = tf.random.normal([9, 8], dtype=tf.float32)
        y = tf.complex(real_part, imag_part)
        y = tf.cast(y, tf.complex64)
        # Call ReciprocalGrad without jit_compile in a normal tf.function context
        result = tf.raw_ops.ReciprocalGrad(dy=x, y=y)
        return result

    @tf.function(jit_compile=True)
    def call_with_jit(self, x):
        # Same as above, but with JIT compilation enabled
        real_part = tf.random.normal([9, 8], dtype=tf.float32)
        imag_part = tf.random.normal([9, 8], dtype=tf.float32)
        y = tf.complex(real_part, imag_part)
        y = tf.cast(y, tf.complex64)
        result = tf.raw_ops.ReciprocalGrad(dy=x, y=y)
        return result

    @tf.function
    def call(self, x):
        # The main __call__ method forwards inputs through both branches and compares the outputs.
        # Since the random y generated inside differs each call, we'll instead feed x as the same tensor for dy,
        # but to mimic the issue example, we must generate y the same way (random).
        # However, this leads to comparing two different outputs from different random ys.
        #
        # To simulate the issue faithfully, we need to generate the same random y for both calls.
        # But random.normal inside separate functions makes the tensors different.
        # To workaround, generate y once here and pass down. This avoids randomness difference.
        #
        # So we modify approach: generate y once outside call_with_jit and call_without_jit.

        real_part = tf.random.normal([9, 8], dtype=tf.float32)
        imag_part = tf.random.normal([9, 8], dtype=tf.float32)
        y = tf.complex(real_part, imag_part)
        y = tf.cast(y, tf.complex64)

        # call_without_jit and call_with_jit need to be rewritten to accept y as argument.
        # For clarity, we adjust to private methods here.

        no_jit_res = self._reciprocal_grad_no_jit(x, y)
        jit_res = self._reciprocal_grad_jit(x, y)

        # Compare the outputs with tolerance as in the issue (atol=0.001, rtol=0.001)
        # Return a bool tensor indicating whether they are near within the tolerance.
        near = tf.debugging.assert_near(
            tf.cast(no_jit_res, tf.float64),
            tf.cast(jit_res, tf.float64),
            atol=0.001,
            rtol=0.001,
            summarize=10
        )
        # tf.debugging.assert_near returns None if no error thrown, so we use tf.reduce_all on abs diff manually
        # to produce a tensor output with difference metric.

        difference = tf.abs(tf.cast(no_jit_res, tf.float64) - tf.cast(jit_res, tf.float64))
        max_diff = tf.reduce_max(difference)
        mean_diff = tf.reduce_mean(difference)

        # Return a dictionary of outputs for inspection:
        # no_jit_res, jit_res, max difference and mean difference tensors.
        return {
            "no_jit_result": no_jit_res,
            "jit_result": jit_res,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "are_close": max_diff < 0.001  # boolean
        }

    @tf.function
    def _reciprocal_grad_no_jit(self, dy, y):
        # Run ReciprocalGrad on CPU without jit_compile
        # We force running eagerly or as normal tf.function on CPU
        # This function is decorated with tf.function (default)
        # but jit_compile is not enabled.
        return tf.raw_ops.ReciprocalGrad(dy=dy, y=y)

    @tf.function(jit_compile=True)
    def _reciprocal_grad_jit(self, dy, y):
        # Run ReciprocalGrad on CPU with jit_compile=True
        return tf.raw_ops.ReciprocalGrad(dy=dy, y=y)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a valid input tensor for reciprocal grad dy
    # From the issue, dy is a complex64 tensor with shape [9,8]
    # constructed from random normal real and imaginary parts
    real_part = tf.random.normal([9, 8], dtype=tf.float32)
    imag_part = tf.random.normal([9, 8], dtype=tf.float32)
    tensor = tf.complex(real_part, imag_part)
    tensor = tf.cast(tensor, tf.complex64)
    return tensor


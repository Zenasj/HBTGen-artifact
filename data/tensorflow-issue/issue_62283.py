# tf.random.normal((8, 10), dtype=tf.float16)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Random matrix 'a' is fixed for the instance to mimic the example's behavior.
        # Shape [10, 8] matches example's tf.random.normal([10, 8], dtype=tf.float16)
        self.a = tf.Variable(
            initial_value=tf.random.normal([10, 8], dtype=tf.float16),
            trainable=False,
            dtype=tf.float16,
        )

    @tf.function(jit_compile=True)
    def call(self, x):
        """
        Performs matmul with jit compilation enabled.
        Returns result.
        """
        # Equivalent to tf.raw_ops.MatMul with transpose_a=False, transpose_b=False
        # Using tf.matmul here for clarity, behavior matches.
        return tf.matmul(self.a, x)

    @tf.function
    def call_nojit(self, x):
        """
        Performs matmul without JIT compilation.
        Returns result.
        """
        return tf.matmul(self.a, x)

    def call_compare(self, x):
        """
        Runs both jit_compile=True and jit_compile=False matmul,
        compares their results elementwise within given tolerance,
        and returns a boolean tensor indicating where they match.

        The tolerances reflect the issue's reported atol=0.001, rtol=0.001.
        """
        # Run jit compiled
        jit_res = self.call(x)
        # Run no jit compiled (eager or no jit)
        nojit_res = self.call_nojit(x)

        # Cast results to float64 for comparison as done in the issue
        jit_res_f64 = tf.cast(jit_res, tf.float64)
        nojit_res_f64 = tf.cast(nojit_res, tf.float64)

        # Compute absolute difference and tolerance mask
        atol = 0.001
        rtol = 0.001
        diff = tf.abs(jit_res_f64 - nojit_res_f64)
        tolerance = atol + rtol * tf.abs(nojit_res_f64)

        # Boolean tensor: True if values close enough, False otherwise
        close = diff <= tolerance

        return close

def my_model_function():
    # Return a new instance of MyModel with random but fixed 'a' matrix
    return MyModel()

def GetInput():
    # Returns a random input tensor consistent with example input:
    # The example uses input shape [8, 10], dtype float16
    # Note in matmul: a [10,8] times x [8,10] -> result [10,10]
    return tf.random.normal([8, 10], dtype=tf.float16)


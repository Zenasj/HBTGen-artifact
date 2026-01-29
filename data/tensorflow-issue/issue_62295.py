# tf.random.normal((9, 8, 6, 3), dtype=tf.bfloat16)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Perform RGB to HSV conversion with jit_compile=True
        return tf.raw_ops.RGBToHSV(images=x)

    def no_jit_call(self, x):
        # Perform RGB to HSV conversion with eager execution (no jit)
        return tf.raw_ops.RGBToHSV(images=x)

    @tf.function
    def call_with_comparison(self, x):
        # Compute both jit-compiled and no-jit results and compare them
        # This fuses the "two models" from the issue and outputs their difference or bool equality.
        no_jit_res = tf.raw_ops.RGBToHSV(images=x)
        # For jit-compiled we wrap in a tf.function with jit_compile=True
        @tf.function(jit_compile=True)
        def jit_func(inp):
            return tf.raw_ops.RGBToHSV(images=inp)
        jit_res = jit_func(x)

        # Cast to float64 for precise comparison as in the issue
        no_jit_res_f64 = tf.cast(no_jit_res, tf.float64)
        jit_res_f64 = tf.cast(jit_res, tf.float64)

        # Compute absolute and relative difference
        abs_diff = tf.abs(no_jit_res_f64 - jit_res_f64)
        rel_diff = abs_diff / (tf.abs(no_jit_res_f64) + 1e-12)

        # Check tolerance, same as issue atol=0.001 and rtol=0.001
        tol_abs = 0.001
        tol_rel = 0.001
        within_tol = tf.math.logical_and(abs_diff <= tol_abs, rel_diff <= tol_rel)

        # Return a boolean tensor indicating comparison per element
        return within_tol, no_jit_res, jit_res

    def call(self, x):
        # The forward call outputs the boolean tolerance tensor,
        # plus both outputs for insight (could also output diff or summary)
        within_tol, nojit_out, jit_out = self.call_with_comparison(x)
        # For usage simplicity, return a dict with all outputs
        return {
            "within_tolerance": within_tol,
            "no_jit_rgb_to_hsv": nojit_out,
            "jit_rgb_to_hsv": jit_out,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching input expected by the model.
    # According to the issue, shape: [9, 8, 6, 3], dtype=tf.bfloat16.
    # Use normal distribution to match original example.
    return tf.random.normal([9, 8, 6, 3], dtype=tf.bfloat16)


# tf.random.uniform((10, 9), dtype=tf.bfloat16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encapsulate the non-JIT and JIT compiled transformations as submodules
        
        # Non-JIT compiled function as a tf.Module function
        # We create a tf.function without jit_compile for reference output
        class NonJITModule(tf.Module):
            @tf.function(jit_compile=False)
            def __call__(self, x):
                x = tf.raw_ops.Sin(x=x)
                x = tf.raw_ops.Acos(x=x)
                return x
        
        # JIT-compiled function as a tf.Module function
        class JITModule(tf.Module):
            @tf.function(jit_compile=True)
            def __call__(self, x):
                x = tf.raw_ops.Sin(x=x)
                x = tf.raw_ops.Acos(x=x)
                return x
        
        self.nonjit = NonJITModule()
        self.jit = JITModule()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Compute both outputs: no_jit and jit compiled versions
        nojit_out = self.nonjit(x)
        jit_out = self.jit(x)

        # Compare outputs using tf.debugging.assert_near equivalent logic but return difference
        # Return a dictionary with outputs and boolean indicating whether results are close within tolerance
        atol = 0.001
        rtol = 0.001

        # Promote to float64 for accurate comparison (consistent with issue code)
        nojit_out_f64 = tf.cast(nojit_out, tf.float64)
        jit_out_f64 = tf.cast(jit_out, tf.float64)

        diff = tf.abs(nojit_out_f64 - jit_out_f64)
        tolerance = atol + rtol * tf.abs(nojit_out_f64)
        is_close = tf.reduce_all(diff <= tolerance)

        # Return dictionary of results and whether close
        # Also return raw difference tensor for inspection
        return {
            "nojit_output": nojit_out,
            "jit_output": jit_out,
            "are_close": is_close,
            "difference": diff,
            "tolerance": tolerance,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the test input shape and dtype from issue:
    # Shape (10, 9), dtype bfloat16
    return tf.random.uniform((10, 9), dtype=tf.bfloat16)


# tf.random.uniform((57, 22), minval=-128, maxval=128, dtype=tf.int8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate Model1 and Model2 logic as submodules
        
    @tf.function(jit_compile=True)
    def call(self, inp):
        """
        Fuse of Model1 and Model2 logic:
        Model1:
          triu = tf.experimental.numpy.triu(inp, k=0)
          reduce_min = tf.math.reduce_min(triu, axis=0)
          output: reduce_min (shape [22])

        Model2:
          triu = tf.experimental.numpy.triu(inp, k=0)
          trans = tf.transpose(triu, perm=[1, 0])  # shape [22, 57]
          concat = tf.concat([trans, trans], axis=0)  # shape [44, 57]
          reduce_min = tf.math.reduce_min(triu, axis=0)  # shape [22]
          output: reduce_min, concat

        This fused model computes both outputs and compares Model1's output (reduce_min)
        against Model2's corresponding reduce_min to detect inequality caused by XLA
        compilation differences on int8 inputs.
        
        Returns:
          A boolean scalar tensor indicating if reduce_min outputs match within atol/rtol tolerances.
          For completeness, also returns both reduce_min tensors and concat output from Model2.
        """
        # Common triu:
        triu = tf.experimental.numpy.triu(inp, k=0)  # shape [57,22], dtype int8
        
        # Model1 reduce_min:
        reduce_min_m1 = tf.math.reduce_min(triu, axis=0)  # shape [22]
        
        # Model2 computations:
        trans = tf.transpose(triu, perm=[1, 0])          # shape [22,57]
        concat = tf.concat([trans, trans], axis=0)       # shape [44,57]
        reduce_min_m2 = tf.math.reduce_min(triu, axis=0) # shape [22]
        
        # Comparison logic:
        # Since dtype is int8, convert to int32 before comparison to avoid overflow issues.
        reduce_min_m1_int = tf.cast(reduce_min_m1, tf.int32)
        reduce_min_m2_int = tf.cast(reduce_min_m2, tf.int32)
        
        # Use absolute tolerance=1 and relative tolerance=0.01 to allow small diff
        abs_diff = tf.abs(reduce_min_m1_int - reduce_min_m2_int)
        rel_diff = abs_diff / (tf.abs(reduce_min_m2_int) + 1e-8)
        
        is_all_close = tf.reduce_all(
            tf.logical_or(abs_diff <= 1, rel_diff <= 0.01)
        )
        
        return is_all_close, reduce_min_m1, reduce_min_m2, concat

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate valid int8 input matching the expected shape [57,22]
    # Uniform int8 range from -128 to 127 inclusive
    # Note: tf.random.uniform maxval is exclusive, hence 128
    inp = tf.random.uniform(
        shape=[57, 22], minval=-128, maxval=128, dtype=tf.int32
    )
    inp = tf.cast(inp, tf.int8)
    return inp


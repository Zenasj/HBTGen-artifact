# tf.add(tf.experimental.numpy.triu(tf.add(inp2, inp1)), axis=1) with inp1 shape=[13,1], inp2 shape=[13,60], dtype=tf.int8

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights, purely operation-based model.
    
    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # inp1: int8 tensor shape [13,1]
        # inp2: int8 tensor shape [13,60]
        # Model1 branch (no transpose output)
        add = tf.add(inp2, inp1)  # Broadcasting add [13,60] + [13,1] -> [13,60]
        triu = tf.experimental.numpy.triu(add, k=0)  # Upper triangular with k=0 on last two dims
        reduce_max = tf.math.reduce_max(triu, axis=1)  # max over axis=1, shape [13]
        
        # Model2 branch (adds transpose output)
        transpose = tf.transpose(add, perm=[1, 0])  # shape [60, 13]
        
        # The original issue compares outputs from both models as a test.
        # Here, we fuse both models and produce a final output that shows 
        # if outputs match within a tolerance, returning their differences.
        
        # To compare triu and reduce_max outputs of both:
        # They actually compute the same triu, reduce_max in both models.
        # Only difference is Model2 has an additional transpose output.
        # The issue showed that under XLA GPU the outputs are inconsistent.
        # We implement both and output a boolean indicating if all close within tolerance.
        
        # To keep compatibility, return a dict with all relevant outputs:
        # triu, reduce_max from model1,
        # triu, reduce_max, transpose from model2,
        # and a boolean tensor indicating close comparison of triu and reduce_max outputs.
        
        # Since both triu and reduce_max in model2 are identical ops to model1,
        # we just replicate them here for demonstration.
        
        # Because triu and reduce_max are tensors (int8), we cast to float32 for numerical comparison.
        triu1 = triu
        reduce_max1 = reduce_max
        
        triu2 = tf.experimental.numpy.triu(add, k=0)
        reduce_max2 = tf.math.reduce_max(triu2, axis=1)
        transpose2 = transpose
        
        # Compare triu outputs and reduce_max outputs element-wise with tolerance
        # Using tf.abs and max element-wise absolute difference
        # Note: Using tf.math.abs and reduce_max element-wise diff.
        
        # Cast to int32 to avoid overflow issues in difference computation
        triu1_int = tf.cast(triu1, tf.int32)
        triu2_int = tf.cast(triu2, tf.int32)
        reduce_max1_int = tf.cast(reduce_max1, tf.int32)
        reduce_max2_int = tf.cast(reduce_max2, tf.int32)
        
        triu_diff = tf.abs(triu1_int - triu2_int)
        reduce_max_diff = tf.abs(reduce_max1_int - reduce_max2_int)
        
        # Define tolerances similarly roughly as rtol=0.001, atol=0.001 for float,
        # For int8 values in range [-128, 127], a tolerance of 1 might be reasonable for demonstration.
        tol = 1
        
        triu_close = tf.reduce_all(triu_diff <= tol)
        reduce_max_close = tf.reduce_all(reduce_max_diff <= tol)
        
        all_close = tf.logical_and(triu_close, reduce_max_close)
        
        # Return dictionary of outputs and comparison result
        return {
            "triu_model1": triu1, 
            "reduce_max_model1": reduce_max1,
            "triu_model2": triu2, 
            "reduce_max_model2": reduce_max2,
            "transpose_model2": transpose2,
            "outputs_match_within_tolerance": all_close
        }

def my_model_function():
    return MyModel()

def GetInput():
    # According to the original code:
    # inp1 is shape [13,1] int8 with values between -128 and 127 inclusive
    # inp2 is shape [13,60] int8 with values between -128 and 127 inclusive
    # Use tf.random.uniform with int32 then cast to int8
    inp1 = tf.cast(
        tf.random.uniform(shape=[13, 1], minval=-128, maxval=128, dtype=tf.int32), 
        tf.int8
    )
    inp2 = tf.cast(
        tf.random.uniform(shape=[13, 60], minval=-128, maxval=128, dtype=tf.int32), 
        tf.int8
    )
    return inp1, inp2


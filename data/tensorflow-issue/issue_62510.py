# tf.random.uniform((10, 10, 10), dtype=tf.int8) ← Input shape inferred from the issue example inputs; scalar input as well

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights, purely functional implementations

    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        """
        Implements the two mathematically equivalent models and compares their outputs.
        
        Model1: (inp1 + inp2)^2 and its absolute value
        Model2: inp1*(inp1 + inp2) + inp2*(inp1 + inp2) and its absolute value
        
        Returns:
            A dictionary with:
            - fin_out_1: Model1 computed result
            - abs_out_1: Model1 absolute output
            - fin_out_2: Model2 computed result
            - abs_out_2: Model2 absolute output
            - outputs_match: boolean tensor indicating if fin_out_1 and fin_out_2 match closely
            - abs_match: boolean tensor indicating if abs_out_1 and abs_out_2 match closely
            - fin_diff: absolute difference between fin_out_1 and fin_out_2
            - abs_diff: absolute difference between abs_out_1 and abs_out_2
        """
        # Model1 computations
        added = tf.add(inp1, inp2)
        fin_out_1 = tf.multiply(added, added)
        abs_out_1 = tf.abs(fin_out_1)

        # Model2 computations
        v5_0 = tf.multiply(inp2, added)
        v6_0 = tf.multiply(inp1, added)
        fin_out_2 = tf.add(v6_0, v5_0)
        abs_out_2 = tf.abs(fin_out_2)

        # Comparison: use relative tolerance of 1e-3 and absolute tolerance of 1e-3 as reported in issue
        rtol = 1e-3
        atol = 1e-3

        fin_close = tf.experimental.numpy.isclose(fin_out_1, fin_out_2, rtol=rtol, atol=atol)
        abs_close = tf.experimental.numpy.isclose(abs_out_1, abs_out_2, rtol=rtol, atol=atol)

        outputs_match = tf.reduce_all(fin_close)
        abs_match = tf.reduce_all(abs_close)

        fin_diff = tf.abs(tf.cast(fin_out_1, tf.float32) - tf.cast(fin_out_2, tf.float32))
        abs_diff = tf.abs(tf.cast(abs_out_1, tf.float32) - tf.cast(abs_out_2, tf.float32))

        return {
            'fin_out_1': fin_out_1,
            'abs_out_1': abs_out_1,
            'fin_out_2': fin_out_2,
            'abs_out_2': abs_out_2,
            'outputs_match': outputs_match,
            'abs_match': abs_match,
            'fin_diff': fin_diff,
            'abs_diff': abs_diff,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    """
    Generate two inputs compatible with MyModel:
    - inp1: scalar int8 tensor random uniform between -128 and 127
    - inp2: int8 tensor of shape (10, 10, 10), random uniform between -128 and 127
    
    This matches the sample inputs used in the issue.
    """
    inp1 = tf.cast(tf.random.uniform(shape=[], minval=-128, maxval=128, dtype=tf.int32), tf.int8)
    inp2 = tf.cast(tf.random.uniform(shape=[10, 10, 10], minval=-128, maxval=128, dtype=tf.int32), tf.int8)
    return inp1, inp2


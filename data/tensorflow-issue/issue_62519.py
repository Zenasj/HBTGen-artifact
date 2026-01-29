# tf.random.uniform((5, 5, 5), dtype=tf.float32) ← input shape inferred from provided example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed, all ops are functional

    @tf.function(jit_compile=True)
    def call(self, inp1):
        # Compute softmax along axis=0 as per example
        softmax = tf.nn.softmax(inp1, axis=0)
        # Transpose softmax output as done in Model2; perm=[0,2,1]
        trans = tf.transpose(softmax, perm=[0, 2, 1])
        # Reduce sum over axis=0 (combines batch dim)
        reduce_sum = tf.math.reduce_sum(trans, axis=0)
        # Cast reduce_sum to int64
        cast = tf.cast(reduce_sum, dtype=tf.int64)

        # Comparison logic:
        # Emulate fusion of Model1 and Model2 behavior.
        # Model1 returns: reduce_sum, cast
        # Model2 returns: reduce_sum, cast, trans
        #
        # The inconsistency reported relates to differences between these outputs
        # under XLA compilation. We will output a dict of all these tensors plus a 
        # boolean tensor indicating elementwise closeness between Model1 and Model2 outputs:
        #
        # Since reduce_sum and cast from both models should be identical, return them once,
        # and also return the transposed tensor. As a proxy for comparison or checking,
        # also return a boolean tensor showing if reduce_sum cast to int64 matches cast from Model1.
        #
        # This mirrors an intended fused model comparing both behaviors internally.
        #
        # Note: The inputs and ops exactly reflect the reported user code, 
        # with comparison logic inferred from the issue context.

        # For demonstration, produce a numeric difference for reduce_sum and cast comparison:
        # Since cast is int64 version of reduce_sum, the difference is reduce_sum - float(cast)
        cast_float = tf.cast(cast, dtype=tf.float32)
        diff_reduce_cast = reduce_sum - cast_float

        return {
            "reduce_sum": reduce_sum,
            "cast": cast,
            "trans": trans,
            "diff_reduce_cast": diff_reduce_cast,
            # Additionally, a boolean tensor showing closeness within atol=1e-3 and rtol=1e-3
            "reduce_cast_close": tf.math.abs(diff_reduce_cast) < 1e-3,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate random tensor input matching shape (5, 5, 5) and dtype float32
    # as per the example inputs in the issue
    return tf.random.uniform(shape=(5,5,5), dtype=tf.float32)


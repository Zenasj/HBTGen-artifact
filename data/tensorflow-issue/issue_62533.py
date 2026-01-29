# tf.random.uniform((15, 1, 50, 35), dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We are fusing Model1 and Model2 from the issue:
        # Model1 returns (greater, logical_and)
        # Model2 returns (greater, logical_and, trans)
        # The forward output here will combine both and perform the comparison
        # between the two sets of outputs to expose the inconsistency described.

    @tf.function(jit_compile=True)
    def call(self, inp):
        # Step 1: Transpose input tensor from (15,1,50,35) to (15,1,35,50)
        trans = tf.transpose(inp, perm=[0, 1, 3, 2])

        # Step 2: Round the transposed tensor
        round_val = tf.round(trans)

        # Step 3: Reverse round_val along axes 0 and 2 (batch and 3rd dim)
        rev_round = tf.reverse(round_val, axis=[0, 2])

        # Step 4: Compute greater boolean tensor as rev_round > round_val
        greater = tf.greater(rev_round, round_val)

        # Step 5: logical_and of greater and greater (redundant but per original code)
        logical_and = tf.logical_and(greater, greater)

        # Model1 outputs: greater, logical_and
        out1 = (greater, logical_and)

        # Model2 outputs: greater, logical_and, trans
        out2 = (greater, logical_and, trans)

        # For comparison:
        # We will compare outputs of Model1 and Model2 for the first two tensors (greater, logical_and)
        # and also check the third tensor from Model2 (trans) matches shape expectations.
        # Return a dictionary with all outputs and a boolean indicating if first two outputs match approx.

        # Since this model fuses them, output both sets plus a diff flag for the first two outputs
        # We use tf.experimental.numpy.isclose for approximate numeric closeness on boolean tensors:
        # Booleans -> cast to int32 for numeric comparison.
        greater_diff = tf.math.reduce_any(tf.not_equal(out1[0], out2[0]))
        logical_and_diff = tf.math.reduce_any(tf.not_equal(out1[1], out2[1]))

        # Return outputs and combined discrepancy flag as single tensor
        # Also return all raw outputs for downstream use / debugging
        # The discrepancy is True if any element differs in either greater or logical_and

        discrepancy = tf.logical_or(greater_diff, logical_and_diff)

        return {
            'model1_outputs': out1,
            'model2_outputs': out2,
            'discrepancy': discrepancy,
        }

def my_model_function():
    # Return an instance of MyModel()
    return MyModel()

def GetInput():
    # Input must match the shape used in the issue: [15,1,50,35]
    # dtype=tf.float64 as in the reproducible code
    return tf.random.uniform(shape=[15, 1, 50, 35], dtype=tf.float64)


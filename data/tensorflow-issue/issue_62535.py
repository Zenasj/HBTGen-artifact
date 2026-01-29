# tf.random.uniform((21, 27, 10, 1, 1), dtype=tf.float32) ← Input shape inferred from issue's test input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constants used in LRN, from the issue
        self.depth_radius = 1
        self.bias = 62.98211185437273
        self.alpha = 22.83989611654185
        self.beta = 0.9124946866870809

    @tf.function(jit_compile=True)
    def call(self, inp, compare=False):
        """
        Runs both submodels internal to this fused model and compares their outputs.
        
        Args:
            inp: input tensor of shape [21, 27, 10, 1, 1], dtype tf.float32
            compare: bool, if True, returns a boolean tensor indicating nearly-equal outputs,
                     accounting for small floating-point differences.
        
        Returns:
            If compare=False: returns a tuple (model1_lrn, model2_lrn, model2_transpose)
            If compare=True: returns a boolean tensor indicating if all outputs are close within tolerance.
        """
        # Both models start by taking cosine of input
        cos = tf.cos(inp)  # shape: (21, 27, 10, 1, 1)

        # Transpose with perm=[4, 1, 2, 3, 0] from the issue
        transpose = tf.transpose(cos, perm=[4, 1, 2, 3, 0])
        # Resulting shape:
        # Original inp shape: [21, 27, 10, 1, 1]
        # cos shape: same as inp
        # transpose permutes dims:
        # dim 0 -> 4, dim 1 -> 1, dim 2 -> 2, dim 3 -> 3, dim 4 -> 0
        # So shape after transpose is: [1, 27, 10, 1, 21]

        # Model1 reduce_min along axis=2
        reduce_min_1 = tf.math.reduce_min(transpose, axis=2)  # shape: (1, 27, 1, 21)

        # Model1 applies LRN on reduce_min_1
        lrn_1 = tf.raw_ops.LRN(input=reduce_min_1,
                              depth_radius=self.depth_radius,
                              bias=self.bias,
                              alpha=self.alpha,
                              beta=self.beta)

        # Model2 reduce_min along axis=2 (same as Model1)
        reduce_min_2 = tf.math.reduce_min(transpose, axis=2)  # same as reduce_min_1

        # Model2 applies LRN on reduce_min_2
        lrn_2 = tf.raw_ops.LRN(input=reduce_min_2,
                              depth_radius=self.depth_radius,
                              bias=self.bias,
                              alpha=self.alpha,
                              beta=self.beta)

        # Model2 also returns transpose as extra output per issue
        # The primary discrepancy arises from having extra output (transpose) and reduce_min usage

        if compare:
            # Compare lrn_1 and lrn_2 with a tolerance matching the issue's np.testing.assert_allclose settings
            # Use atol=0.001, rtol=0.001 as in the issue
            are_close = tf.reduce_all(tf.math.abs(lrn_1 - lrn_2) <= 0.001 + 0.001 * tf.math.abs(lrn_2))
            return are_close
        else:
            return lrn_1, lrn_2, transpose

def my_model_function():
    # Creates the model instance
    return MyModel()

def GetInput():
    # Returns a single input tensor shaped exactly as in the issue.
    # dtype float32 and shape (21, 27, 10, 1, 1)
    return tf.random.uniform(shape=[21, 27, 10, 1, 1], dtype=tf.float32)


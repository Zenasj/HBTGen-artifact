# tf.random.uniform((7, 5, 49, 1, 1), dtype=tf.float32) ← inferred input shape from usage in original code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constant parameter tensor used in division (shape [49, 9, 1])
        self.p0 = tf.constant(tf.random.uniform(shape=[49, 9, 1], dtype=tf.float32))

    @tf.function(jit_compile=True)
    def call(self, inp):
        """
        This fused model encapsulates the logic of Model1 and Model2 from the issue.
        Inputs:
          - inp: float32 tensor shaped [7, 5, 49, 1, 1]

        Outputs:
          - A boolean tensor indicating elementwise closeness of outputs between
            the two models (runtimes under XLA jit compilation differ subtly).
        
        Explanation:
          Model1 returns (red, cos)
          Model2 returns (red, cos, transposed_concat)
          where
            red = reduce_prod(transposed_div, axis=2)
            cos = cos(red)
            transposed_div = transpose(div, perm=[0,1,3,2,4])
            div = tf.divide(inp, p0)

          Model2 adds:
            concat = concat([transposed_div, transposed_div], axis=2)
            transposed_concat = transpose(concat, perm=[1,0,2,3,4])

          This fusion runs both computations and compares their outputs (red and cos),
          ignoring transposed_concat for comparison as Model1 does not produce it.

        Returns:
          - A boolean tensor indicating closeness of (red1 vs red2) and (cos1 vs cos2)
            with specified tolerances.
        """

        # Shared computation for division and transpose
        div = tf.divide(inp, self.p0)  # Broadcasting division, inp shape [7,5,49,1,1], p0 shape [49,9,1]
        transposed_div = tf.transpose(div, perm=[0, 1, 3, 2, 4])  # shape: [7,5,1,49,1]

        # Model1 outputs
        red1 = tf.math.reduce_prod(transposed_div, axis=2)  # reduce over axis=2 (which is the 1 in shape)
        cos1 = tf.cos(red1)

        # Model2 outputs
        concat = tf.concat([transposed_div, transposed_div], axis=2)  # axis=2 doubles size along that axis
        transposed_concat = tf.transpose(concat, perm=[1, 0, 2, 3, 4])
        red2 = tf.math.reduce_prod(transposed_div, axis=2)  # same as Model1
        cos2 = tf.cos(red2)

        # Compare outputs of Model1 and Model2 (red and cos)
        # Using elementwise closeness with tolerances used in the issue
        red_close = tf.math.abs(red1 - red2) <= 0.001 + 0.001 * tf.math.abs(red2)
        cos_close = tf.math.abs(cos1 - cos2) <= 0.001 + 0.001 * tf.math.abs(cos2)

        # Final boolean output: True where all elements close, False otherwise
        outputs_close = tf.logical_and(red_close, cos_close)

        return outputs_close

def my_model_function():
    # Return an instance of the fused model
    return MyModel()

def GetInput():
    # Return a random tensor input compatible with MyModel
    # According to the original code shape: [7, 5, 49, 1, 1], dtype float32
    return tf.random.uniform(shape=[7, 5, 49, 1, 1], dtype=tf.float32)


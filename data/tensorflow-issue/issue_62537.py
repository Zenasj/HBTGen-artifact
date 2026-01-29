# tf.random.uniform((B, 8), dtype=tf.float64) and tf.random.uniform((61, 4), dtype=tf.float64) 
# Input shapes inferred from example: Four inputs of shape [1,8] and one input of shape [61,4]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights or layers - purely functional model as per original code

    @tf.function(jit_compile=True)
    def call(self, inp1, inp2, inp3, inp4, inp5):
        """
        Compose a fused model combining behavior of Model1 and Model2 from the issue:
        - Concatenate inp4, inp3, inp2, inp1 along axis=0
        - Compute matmul of inp5 and concat
        - Compute transpose of matmul
        - Return:
          - output from Model1: (out, extra_trans) as tuple of two identical transposes
          - output from Model2: (out,) single transpose output

        Then compare these outputs for consistency.
        The forward output is a boolean indicating if all outputs match within tolerances.

        This fused behavior aligns with the issue's discussion about differences between
        Model1 (which returns two transpose outputs) and Model2 (returns only one).
        """

        concat = tf.concat([inp4, inp3, inp2, inp1], axis=0)  # Shape: (4,8)
        matmul = tf.matmul(inp5, concat)  # inp5 (61,4) @ concat (4,8) => (61,8)
        out = tf.transpose(matmul, perm=[1, 0])  # (8,61)
        extra_trans = tf.transpose(matmul, perm=[1, 0])  # identical transpose

        # Outputs as in original models
        model1_outputs = (out, extra_trans)
        model2_outputs = (out,)

        # Compare outputs elementwise within tolerances
        # For simplicity, we focus on the difference between model1_outputs[0] and model2_outputs[0]
        # Additionally, check if model1_outputs[0] == model1_outputs[1] to verify identical outputs

        rtol = 1e-3
        atol = 1e-3

        # Check if model1's two outputs are close: they should be identical
        model1_outputs_close = tf.reduce_all(
            tf.math.abs(model1_outputs[0] - model1_outputs[1]) <= atol + rtol * tf.math.abs(model1_outputs[1])
        )
        # Check if model1 output compared to model2 output are close
        model1_model2_close = tf.reduce_all(
            tf.math.abs(model1_outputs[0] - model2_outputs[0]) <= atol + rtol * tf.math.abs(model2_outputs[0])
        )

        # Final output: True if both close checks pass, False otherwise
        comparison_result = tf.logical_and(model1_outputs_close, model1_model2_close)

        # Return a dictionary with all outputs and comparison result for clarity
        return {
            "model1_out": model1_outputs[0],
            "model1_extra_trans": model1_outputs[1],
            "model2_out": model2_outputs[0],
            "are_model1_outputs_close": model1_outputs_close,
            "are_model1_and_model2_outputs_close": model1_model2_close,
            "overall_comparison_passed": comparison_result,
        }

def my_model_function():
    # Return an instance of MyModel without weights initialization (stateless)
    return MyModel()

def GetInput():
    # Return the exact input tuple used in the issue:
    # Four inputs each [1,8] and one input [61,4], dtype=tf.float64
    inp1 = tf.random.uniform(shape=[1, 8], dtype=tf.float64)
    inp2 = tf.random.uniform(shape=[1, 8], dtype=tf.float64)
    inp3 = tf.random.uniform(shape=[1, 8], dtype=tf.float64)
    inp4 = tf.random.uniform(shape=[1, 8], dtype=tf.float64)
    inp5 = tf.random.uniform(shape=[61, 4], dtype=tf.float64)
    return (inp1, inp2, inp3, inp4, inp5)


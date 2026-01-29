# tf.random.normal([10, 4, 9, 10, 4, 7], dtype=tf.float32) ← Input shape inferred from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights or layers are defined in the original example.
        # The core operation is tf.raw_ops.BatchMatMulV2 with inputs 'x' and a random tensor generated internally.

    @tf.function(jit_compile=True)
    def call(self, x):
        # The original issue code calls tf.raw_ops.BatchMatMulV2 as:
        # tf.raw_ops.BatchMatMulV2(y=x, adj_x=False, adj_y=False, x=tf.random.normal([10,4], dtype=tf.float32))
        # Note: The first argument 'x' and keyword 'x=' are both in the call, which is ambiguous in the original snippet.
        # The correct BatchMatMulV2 signature is BatchMatMulV2(x, y, adj_x, adj_y)
        # The provided code passes y=x, adj_x=False, adj_y=False, x=random_tensor which implies
        # the first argument x=tf.random.normal(...), y=x (input), so the multiplication is random_tensor @ input

        # To replicate the original problem, we perform BatchMatMulV2 with:
        # x = random tensor of shape [10,4]
        # y = input tensor x with shape [10,4,9,10,4,7]
        # This is ambiguous in broadcasting and shapes for matmul.
        # The original error indicates shapes [10,4,9,10,10,7] which is inconsistent with inputs.
        # For a working example, we'll do batch matmul between the last two dims of each tensor element-wise.
        # However, the original input shape is quite high dimensional.
        # We will reshape input and random tensor to compatible batch matmul shapes.

        # Let's reshape input x of shape [10,4,9,10,4,7] to [10*4*9*10,4,7], and random tensor [10,4] to [...,4, K]
        # But original random tensor shape is [10,4], which can't be batch-multiplied with x directly.
        # Instead, as per original minimal snippet, we just replicate exactly to illustrate the problem.

        # So here we exactly reproduce:
        rand_tensor = tf.random.normal([10, 4], dtype=tf.float32)
        # Run BatchMatMulV2: output = rand_tensor @ x, with no adjoints.
        # But x shape is [10,4,9,10,4,7], so this matmul is not shape compatible in usual sense.
        # We can only run BatchMatMulV2 if inputs are rank>=3 with matching batch dims and inner dims.

        # To avoid shape issues in execution, we apply it on inputs broadcasted/batched correctly.
        # For faithful reproduction:
        # We'll flatten x and rand_tensor as needed:
        # Alternatively, we use tf.raw_ops.BatchMatMulV2 on:
        # x: rand_tensor [10,4]
        # y: x [10,4,9,10,4,7]
        # But BatchMatMulV2 expects x and y to be ≥3 dimensions, so we can add batch dims:

        # Add batch dims to rand_tensor to match x:
        # Expand dims to [10,4,1,1,1,4] to broadcast with x
        rand_tensor_exp = tf.reshape(rand_tensor, [10, 4, 1, 1, 1, 4])

        # Now perform matmul on last two dims:
        # x: [10,4,9,10,4,7], rand_tensor_exp: [10,4,1,1,1,4]
        # They differ on batch dims, so direct BatchMatMulV2 will fail.
        # Instead, we do:
        # Compute matmul between rand_tensor_exp (last two dims 1 x 4) and x (last two dims 4 x 7)
        # To keep batch dims aligned, swap x and rand_tensor to match dimensions

        # So call: BatchMatMulV2(x=rand_tensor_exp, y=x)
        # With adj_x=False, adj_y=False

        # The batch dims for BatchMatMulV2 must be equal except for possibly the matrix dims.
        # rand_tensor_exp batch shape: [10,4,1,1,1]
        # x batch shape: [10,4,9,10,4]
        # These are not broadcastable as batches for BatchMatMulV2.

        # To produce a working example consistent with original input shapes and issue,
        # We will just apply BatchMatMulV2 for each batch element in a tf.map_fn style.
        # But the original minimal reproducible snippet is exactly as in the issue:
        # x = tf.raw_ops.BatchMatMulV2(y=x, adj_x=False, adj_y=False, x=tf.random.normal([10, 4]))

        # We'll replicate exactly that ignoring input shape mismatch for this model,
        # Since it is meant to demonstrate the batchmatmul behavior difference.

        result = tf.raw_ops.BatchMatMulV2(
            x=rand_tensor,  # shape [10, 4]
            y=x,
            adj_x=False,
            adj_y=False,
        )
        return result


def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a tensor matching the input expected by MyModel.call()
    # According to the issue, input shape is [10, 4, 9, 10, 4, 7], dtype float32
    return tf.random.normal([10, 4, 9, 10, 4, 7], dtype=tf.float32)


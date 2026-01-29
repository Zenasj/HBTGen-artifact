# tf.random.normal((8, 3, 8, 7, 3, 4), dtype=tf.float32) ← inferred input shape from issue repro code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The raw ops BatchMatMulV2 and Cos are used directly as in the issue example.
        # The BatchMatMulV2 op requires two inputs x and y; in the original snippet,
        # x is provided as input tensor, y is a tf.random.normal tensor.
        # We'll keep the same logic here for consistency.

    @tf.function(jit_compile=True)
    def call(self, x):
        # x shape: [8, 3, 8, 7, 3, 4]
        # We need to apply BatchMatMulV2 with x as one input, and another tensor y as the other.
        # Original code calls BatchMatMulV2(y=x, x=random_tensor)
        # But that is non-intuitive; original code snippet uses
        # tf.raw_ops.BatchMatMulV2(y=x, x=random.normal(...))
        # i.e. x=random normal tensor with shape [8,3], y=input tensor.
        # However BatchMatMulV2 expects batches of matrices for x and y.

        # For the sake of reasonable inference and replicating the issue behavior,
        # we recreate a compatible random tensor with shape matching the last two dims:
        # We assume last two dims are matrices to be multiplied.

        # From the original input shape: [8,3,8,7,3,4], presumably:
        # Batch dims: [8,3,8,7,3], Mat dims: (4)
        # But BatchMatMulV2 expects input shape [..., M, K] and [..., K, N], outputs [..., M, N]

        # Let's reduce complexity by flattening some dims:
        # Flatten first 5 dims as batch, last two as matrices:
        batch_shape = tf.shape(x)[:-2]  # [8,3,8,7,3]
        x_matrices = x  # shape [..., 3,4]

        # Generate a random tensor for the 'x' input of BatchMatMulV2 with shape [..., 4,5] to match multiplication
        # So that the multiplication shape is [..., 3,4] @ [..., 4,5] = [..., 3,5]
        # This is reasonable to produce an output

        x_random = tf.random.normal(tf.concat([batch_shape, [4, 5]], axis=0), dtype=tf.float32)

        # Do BatchMatMulV2: x_random @ x_matrices, result shape [..., 4, 4]? 
        # Actually, with inputs:
        # x: x_random -> [...,4,5]
        # y: x -> [...,3,4]
        # For multiplication, last dims should be [..., M,K] and [..., K,N] => output [..., M, N]

        # Let's follow the original call: BatchMatMulV2(y=x, x=random).
        # So y is input tensor x with shape [..., 3,4]
        # x is random normal with shape matching [..., 4, something], so that matmul is valid
        # So x_random shape: [..., 8,3]? That wouldn't multiply.

        # To keep dimension consistent and multiplication valid:
        # Suppose we want to multiply random tensor of shape [..., 4, K] @ input x of shape [..., 3, 4]
        # So x_random shape [..., K, 3] and x shape [..., 3, 4] => not consistent.

        # Actually, in BatchMatMulV2:
        # output = matmul(x, y)

        # In code: tf.raw_ops.BatchMatMulV2(y=x, x=random_tensor)
        # So shapes:
        # x: [?, M, K]
        # y: [?, K, N]
        # Output: [?, M, N]

        # Given input x has shape [..., 3, 4], it must act as y with shape [..., K=3, N=4]
        # So x_random must be [..., M, K=3]

        # Let's create x_random shape [..., M, 3], say M=5
        x_random = tf.random.normal(tf.concat([batch_shape, [5, 3]], axis=0), dtype=tf.float32)
        # Now result shape is [..., 5, 4]

        # Perform BatchMatMulV2 with x_random and x:
        x = tf.raw_ops.BatchMatMulV2(x=x_random, y=x, adj_x=False, adj_y=False)

        # Then apply Cos operation element-wise:
        x = tf.raw_ops.Cos(x=x)
        return x

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Provide an input tensor matching shape expected by MyModel.call(x)
    # From issue: input tensor shape is [8, 3, 8, 7, 3, 4], dtype float32
    return tf.random.normal([8, 3, 8, 7, 3, 4], dtype=tf.float32)


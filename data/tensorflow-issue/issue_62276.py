# tf.random.normal((7, 8), dtype=tf.float32) ← inferred input shape from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Pre-generate y tensor so that the same y is used for every call,
        # addressing the issue where y was generated inside call causing nondeterministic output.
        # The shape and dtype for y is inferred from the sample code and error logs.
        # Note: The original y shape in the example code was a 6D tensor [7, 8, 8, 8, 8, 2].
        self.y = tf.random.normal([7, 8, 8, 8, 8, 2], dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, x):
        # Perform batch matmul with fixed y
        x = tf.raw_ops.BatchMatMulV2(x=x, adj_x=False, adj_y=False, y=self.y)
        x = tf.raw_ops.Sin(x=x)
        return x

def my_model_function():
    # Instantiate and return the MyModel instance with y fixed at init
    return MyModel()

def GetInput():
    # Return random input tensor matching expected input for BatchMatMulV2
    # From raw_ops.BatchMatMulV2 documentation and sample code, x must be shape [..., M, K]
    # y is [7,8,8,8,8,2]; so last two dims are 8 x 2 => K=8, N=2 for y
    # The batch dims need to match leading dims before M,K

    # In the example, x shape used is [7,8] -- but that shape is ambiguous for BatchMatMulV2.
    # BatchMatMulV2 expects x shape [..., M, K] and y shape [..., K, N]
    # Given y shape [7, 8, 8, 8, 8, 2], last two dims are (8,2) => K=8, N=2
    # So x last two dims must be (..., M, 8).
    # The leading dims of x and y (except last two dims) should broadcast or match (for batch matmul).

    # The minimal consistent shape for x to match y=[7,8,8,8,8,2] is:
    # x shape = [7, 8, 8, 8, 8, M=8] so matmul of (batch..., M=8, K=8) @ (batch..., K=8, N=2)
    # But the original code uses shape [7,8], which is not compatible for BatchMatMulV2 as used.

    # To enable the matmul, make x shape [7, 8, 8, 8, 8, 8] - last two dims (8,8)
    # Then x @ y: (7,8,8,8,8,8) times (7,8,8,8,8,2)
    # This matches K=8 between x and y, resulting in output shape (7,8,8,8,8,2)

    # We'll generate such an input here consistent with y shape:
    return tf.random.normal([7, 8, 8, 8, 8, 8], dtype=tf.float32)


# tf.random.normal((9, 8, 6), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Generate a random tensor "tensor" of shape [9,8,6] for RealDiv denominator
        tensor = tf.random.normal([9, 8, 6], dtype=tf.float32)
        # Perform RealDiv operation: y=x, x=tensor (i.e. tensor denominator)
        realdiv_res = tf.raw_ops.RealDiv(y=x, x=tensor)

        # Generate another random tensor "tensor1" of shape [8, 1] for Zeta op parameter q
        tensor1 = tf.random.normal([8, 1], dtype=tf.float32)

        # Perform Zeta operation with q=realdiv_res, x=tensor1
        zeta_res = tf.raw_ops.Zeta(q=realdiv_res, x=tensor1)

        return zeta_res

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching expected shape for 'y' in RealDiv.
    # Since tensor in RealDiv is [9,8,6], y must be broadcastable to that shape.
    # The example used input of shape [1], so we will provide a shape (9,8,6) tensor to avoid broadcasting issues.

    # Assumption: x should broadcast to shape [9,8,6], so make it (9,8,6)
    # We keep dtype float32 to match operations' dtype.
    return tf.random.normal([9, 8, 6], dtype=tf.float32)


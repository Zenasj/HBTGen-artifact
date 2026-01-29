# tf.random.normal((6, 10, 10, 1), dtype=tf.float32) ← input shape inferred from how tensor is used in the original call

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # The operation:
        # original tensor (called `tensor` in the issue) is given as input x and used as raw_ops.Zeta q input
        # The issue originally generated the random tensor inside the call, which caused nondeterminism in XLA.
        # We'll assume input x matches that "tensor".
        # Then compute zeta(q=x, x=tensor_input) with tensor_input = x itself.
        # From the original code the signature is: tf.raw_ops.Zeta(q=..., x=...)
        # In the issue example q=x_input, x=tensor(random normal)
        
        # The original code used:
        # x = tf.raw_ops.Zeta(q=x, x=tensor)
        # then x = tf.raw_ops.Cosh(x=x)
        # return x
        
        # So to keep the same logic:
        zeta_result = tf.raw_ops.Zeta(q=x, x=x)  # Using same tensor for q and x as input for deterministic behavior
        cosh_result = tf.raw_ops.Cosh(x=zeta_result)
        return cosh_result

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # According to the original issue, the random normal tensor passed as input to the model's call is of shape `[6, 10, 10, 1, 8]`
    # But inside the call, tensor is shape `[10, 10, 6, 1]` - likely the op shapes are broadcastable or reordered.
    #
    # The actual tf.raw_ops.Zeta requires "q" and "x" to be broadcast compatible.
    # The original input shape for "q" was `[6,10,10,1,8]`, "x" was `[10,10,6,1]`.
    # To keep it consistent and simpler for the fused model, let's input a shape that matches the largest dimension with float32.
    #
    # Because MyModel uses the input tensor both as q and x in the op, shapes must be broadcast compatible.
    # Let's choose shape consistent with original input q: [6,10,10,1,8]
    return tf.random.normal(shape=(6, 10, 10, 1, 8), dtype=tf.float32)


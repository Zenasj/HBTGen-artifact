# tf.random.uniform((10, 1, 5, 10), dtype=tf.float32) ← Input shape inferred from the example usage in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # In the original issue, a random tensor was generated inside the call causing nondeterminism under jit_compile.
        # To reflect the intended operation and allow stable behavior, we accept x as input and use a fixed "random_tensor"
        # generated deterministically outside or from x's shape.
        # However, since the original issue centers around tf.raw_ops.Zeta behavior, we'll replicate a similar structure here.
        
        # We'll create a deterministic tensor shaped similarly to the issue's "random_tensor" for consistency.
        # Because the original problem showed inconsistency with random normal on jit_compile, 
        # here, we create a fixed tensor of ones for stable output.
        random_tensor = tf.ones([2, 1, 10], dtype=tf.float32)
        
        # Use tf.raw_ops.Zeta with q=x and x=random_tensor as in the original snippet.
        # Note: input "x" shape is (10,1,5,10) but tf.raw_ops.Zeta expects q and x to be broadcastable.
        # The original code used random_tensor shape (2, 1, 10) and x shape (10,1,5,10),
        # which likely relies on broadcasting. We'll keep the same shapes as input.
        result = tf.raw_ops.Zeta(q=x, x=random_tensor)
        
        return result

def my_model_function():
    # Returns an instance of MyModel with fixed behavior.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape used in the issue:
    # Shape: (10, 1, 5, 10), dtype float32
    return tf.random.uniform((10, 1, 5, 10), dtype=tf.float32)


# tf.random.normal((9, 9, 7, 10, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The Network module in the issue runs two raw ops:
        # tf.raw_ops.Zeta and tf.raw_ops.Square in sequence
        # We replicate the same here.
        # Note: We assume input 'x' has shape [7, 10, 8] as in example.
    
    @tf.function(jit_compile=True)
    def call(self, x):
        # Generate fixed tensor of shape [9,9,7,10,1], float32 as in original code
        tensor = tf.random.normal([9, 9, 7, 10, 1], dtype=tf.float32)
        
        # Perform tf.raw_ops.Zeta with q=x, and x=tensor
        zeta_out = tf.raw_ops.Zeta(q=x, x=tensor)
        
        # Then square the output
        squared = tf.raw_ops.Square(x=zeta_out)
        return squared

def my_model_function():
    return MyModel()

def GetInput():
    # The input 'x' provided to tf.raw_ops.Zeta should match shape compatible
    # with tensor argument shape [9,9,7,10,1] for broadcasting rules.
    # Original code generated tensor shape [7,10,8] for input x.
    # We'll keep the input shape as [7, 10, 8] dtype float32 to match example.
    # This matches the issue example, where input shape was [7,10,8].
    return tf.random.normal([7, 10, 8], dtype=tf.float32)


# tf.random.uniform((1, 30, 5), dtype=tf.float32) ← input shape inferred from provided examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, squeeze=False):
        super().__init__()
        self.squeeze = squeeze
        self.dense = tf.keras.layers.Dense(5)
    
    def call(self, x):
        # Squeeze input tensor only if specified to reduce rank from 3 to 2 before Dense,
        # since the bug described is affected by input rank for fully connected layers.
        if self.squeeze:
            # The squeeze here removes dimensions of size 1:
            # input shape example: [1, 30, 5] → squeeze → [30, 5]
            # which makes the Dense layer behave properly with rank-2 input.
            x = tf.squeeze(x)
        return self.dense(x)

def my_model_function():
    # By default, create model without squeezing, matching original input shape [1,30,5].
    # Users can instantiate with squeeze=True if desired to use the rank 2 input workaround.
    return MyModel(squeeze=False)

def GetInput():
    # Generate a random input tensor of shape (1, 30, 5) with float32 dtype,
    # matching the input shape used in the original reproducer.
    return tf.random.uniform(shape=(1, 30, 5), dtype=tf.float32)


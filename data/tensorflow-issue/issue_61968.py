# tf.random.uniform((1, 1, 2), dtype=tf.float32) ← inferred input shape from the original code

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A simple variable like in the original example (not really used in call)
        self.b = tf.Variable(np.array([[1],[2]], dtype=np.float32))

    def call(self, x):
        # x shape: (1, 1, 2)
        # Add 1 as in original model
        x = tf.add(x, 1)
        # Transpose and then L2 normalize
        # Original code uses tf.transpose(x), which transposes the last 2 dims implicitly as it was 3D input
        # TensorFlow's tf.transpose with default perm reverses all dims -> (2, 1, 1)
        x = tf.transpose(x)
        # Normalize l2 along axis=0 is consistent with L2 norm of vectors along first axis (from PyTorch example)
        # The original TF code uses tf.math.l2_normalize, which normalizes along last axis by default,
        # but because shape changes after transpose, we keep default axis=0 to replicate original behavior.
        x = tf.math.l2_normalize(x, axis=0)
        return x

def my_model_function():
    # Simply return a new instance of the model without loading any weights
    return MyModel()

def GetInput():
    # Return a tensor shaped like the original example input: [1, 1, 2]
    # Use tf.random.uniform with float32 dtype as a valid input
    return tf.random.uniform((1, 1, 2), dtype=tf.float32)


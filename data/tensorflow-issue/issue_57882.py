# tf.constant(True, shape=[2,2,2], dtype=tf.bool) ← inferred input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constant tensor of shape [1,1,2,2,1] with bool type for XOR operation
        self.const = tf.constant(True, shape=[1,1,2,2,1], dtype=tf.bool)

    @tf.function
    def call(self, x):
        # Perform logical XOR between input x and self.const
        y = tf.math.logical_xor(x, self.const)
        # Squeeze the first axis as in original code (axis=0)
        y = tf.squeeze(y, axis=0)
        return y

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return a bool tensor matching the input expected by MyModel:
    # The example input was shape [2,2,2], dtype bool
    return tf.constant(True, shape=[2,2,2], dtype=tf.bool)


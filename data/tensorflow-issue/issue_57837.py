# tf.constant(True, shape=[2], dtype=tf.bool) → input shape: (2,) with dtype=tf.bool

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constant boolean tensor of shape [2,2], all True
        self.const = tf.constant(True, shape=[2, 2], dtype=tf.bool)

    @tf.function
    def call(self, x):
        # Apply logical OR between constant and input tensor (broadcasting input of shape [2])
        x = tf.logical_or(self.const, x)
        # Reshape result to [2, 2, 1, 1]
        x = tf.reshape(x, [2, 2, 1, 1])
        return x

def my_model_function():
    # Returns an instance of MyModel, ready to use.
    return MyModel()

def GetInput():
    # Return a boolean tensor matching expected input shape (2,) with all True values.
    # Matches example in original code: tf.constant(True, shape=[2], dtype=tf.bool)
    return tf.constant([True, True], shape=[2], dtype=tf.bool)


# tf.random.uniform((B, 5), dtype=tf.float64), tf.random.uniform((B, 6), dtype=tf.float64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights or layers needed here; just concatenation
        # Inputs dtype set explicitly to float64 to avoid dtype mismatch errors

    def call(self, inputs):
        # inputs is expected to be a list or tuple of two tensors:
        # inputs[0]: shape (batch_size, 5), dtype float64
        # inputs[1]: shape (batch_size, 6), dtype float64
        x1, x2 = inputs
        # Concatenate along last axis (axis=1)
        # Inputs and output maintain dtype float64 consistently
        return tf.concat([x1, x2], axis=1)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate sample inputs matching the expected dtype and shapes
    # Using batch size = 2 for example
    B = 2
    x1 = tf.random.uniform((B, 5), dtype=tf.float64)
    x2 = tf.random.uniform((B, 6), dtype=tf.float64)
    return [x1, x2]


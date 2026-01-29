# tf.random.uniform((B,), dtype=tf.int32)  ← Input shape: scalar int32 (the parameter n)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the Fibonacci computation from the issue example,
    implemented as a subclassed Keras model with a single scalar integer input `n`.
    It uses a loop and two TensorArrays of different types (float32 and int32),
    accumulating the Fibonacci sequences, then returns their sum as a float32 tensor.
    """

    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, n):
        # n is scalar int32 tensor representing the length of the Fibonacci sequence to compute

        ta = tf.TensorArray(tf.float32, size=n)
        tb = tf.TensorArray(tf.int32, size=n)
        ta = ta.write(0, 0.)
        ta = ta.write(1, 1.)
        tb = tb.write(0, 0)
        tb = tb.write(1, 1)

        for i in tf.range(2, n):
            val_float = ta.read(i - 1) + ta.read(i - 2)
            val_int = tb.read(i - 1) + tb.read(i - 2)
            ta = ta.write(i, val_float)
            tb = tb.write(i, val_int)

        result = ta.stack() + tf.cast(tb.stack(), dtype=tf.float32)
        return result

def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a valid scalar int32 tensor input for MyModel
    # Choosing n=10 as a reasonable default input size >= 2 for Fibonacci sequence length
    return tf.constant(10, dtype=tf.int32)


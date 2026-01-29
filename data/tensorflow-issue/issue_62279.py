# tf.random.normal((10, 9), dtype=tf.bfloat16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply cosine followed by square using raw_ops as per the original issue
        x = tf.raw_ops.Cos(x=x)
        x = tf.raw_ops.Square(x=x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor with shape [10, 9] and dtype bfloat16,
    # matching the original example input.
    return tf.random.normal([10, 9], dtype=tf.bfloat16)


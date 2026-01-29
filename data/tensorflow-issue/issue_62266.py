# tf.complex tensor input with shape (8, 2), dtype=tf.complex64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Generate a random complex scalar tensor each call to mimic original randomness
        real_part = tf.random.normal([1], dtype=tf.float32)
        imag_part = tf.random.normal([1], dtype=tf.float32)
        tensor = tf.complex(real_part, imag_part)
        tensor = tf.cast(tensor, dtype=tf.complex64)

        # Note: tf.raw_ops.SquaredDifference expects 'x' and 'y' and computes (x - y)^2 element-wise
        # We swap input order as in original code: y=x, x=tensor
        # Compute squared difference between input tensor and random scalar tensor
        sq_diff = tf.raw_ops.SquaredDifference(y=x, x=tensor)  # (x - tensor)^2 

        # Apply cosine element-wise on squared difference tensor
        cos_res = tf.raw_ops.Cos(x=sq_diff)

        return cos_res

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random complex64 tensor of shape (8, 2) matching expected input
    real_part = tf.random.normal([8, 2], dtype=tf.float32)
    imag_part = tf.random.normal([8, 2], dtype=tf.float32)
    tensor = tf.complex(real_part, imag_part)
    tensor = tf.cast(tensor, dtype=tf.complex64)
    return tensor


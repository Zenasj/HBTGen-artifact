# tf.random.uniform((9,), minval=0, maxval=255, dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters, just a raw op call
        # We'll reuse the random tensor generation inside call for parity with example

    @tf.function(jit_compile=True)
    def call(self, x):
        # Random int32 tensor of shape [9], values in [0, 255)
        random_tensor = tf.random.uniform([9], minval=0, maxval=255, dtype=tf.int32)
        int8_tensor = tf.cast(random_tensor, tf.int8)
        # Apply raw LeftShift as in the issue repro
        shifted = tf.raw_ops.LeftShift(y=x, x=int8_tensor)
        return shifted

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # According to the issue repro, input 'x' is an int8 tensor scalar in example,
    # but model expects x to be shape (9,) for shifting with shape (9,) shift amount.
    # The original code uses shape [] scalar int8 cast from int32 tensor,
    # but then passes that scalar as x to LeftShift with the other tensor as y.
    # The code snippet had: tf.raw_ops.LeftShift(y=x, x=int8_tensor) where x is input,
    # note parameter names in raw_ops.LeftShift: x is shift amount, y is value to shift.

    # Reconstructing consistent input based on usage:
    # Let's make input tensor shape (9,), dtype int8, as reserve for value to be shifted.
    # This allows broadcasting consistent with the shift tensor of shape (9,).

    # Generate a random int8 tensor of shape [9], values from -128 to 127
    # Because casting from int32 [0,255) can saturate or wrap.

    rand_int32 = tf.random.uniform([9], minval=0, maxval=256, dtype=tf.int32)
    int8_tensor = tf.cast(rand_int32, tf.int8)
    return int8_tensor


# tf.random.uniform((4, 1), dtype=tf.int8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No extra layers needed, this model just performs a RightShift op inside a jit-compiled function
        
    @tf.function(jit_compile=True)
    def call(self, x):
        # Mimics the reported buggy behavior using tf.raw_ops.RightShift with jit_compile=True
        # Random shift amount in int8 scalar (simulated here as a fixed shift for determinism in this example)
        # The original issue used a random scalar, but for stability in execution we use a fixed shift amount
        shift_amount = tf.constant(2, dtype=tf.int8)  # Shift by 2 bits as a deterministic example

        # Perform right shift using tf.raw_ops.RightShift where y (input) is shifted right by x (shift_amount)
        # Note: tf.raw_ops.RightShift(y=x, x=shift_amount) means output = y >> x
        shifted = tf.raw_ops.RightShift(y=x, x=shift_amount)
        return shifted

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Provide an input tensor of shape (4, 1), dtype int8, values in 0-255 range cast to int8
    # This matches the example input used in the issue: 
    # random int32 tensor [4,1] in [0,255), then cast to int8
    random_int32 = tf.random.uniform([4,1], minval=0, maxval=255, dtype=tf.int32)
    input_int8 = tf.dtypes.cast(random_int32, tf.int8)
    return input_int8


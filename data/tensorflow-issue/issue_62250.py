# tf.constant(scalar, dtype=tf.complex128) ← Input is a complex128 scalar tensor

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)  # mimic the JIT compilation context of the issue
    def call(self, x):
        # Apply Acos then Exp raw ops, as per issue repro code
        x = tf.raw_ops.Acos(x=x)
        x = tf.raw_ops.Exp(x=x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a complex128 scalar tensor to match the input the model expects
    # Using the exact complex number from the original issue repro code
    ele = -731778.6211090556 - 59304.1731637927j
    return tf.constant(ele, dtype=tf.complex128)


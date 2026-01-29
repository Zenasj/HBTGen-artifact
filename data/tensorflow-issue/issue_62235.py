# tf.random.normal((10, 9, 8), dtype=tf.float64) used to create inputs and subcomponents

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters; this model just wraps the ops in one callable
     
    @tf.function(jit_compile=True)
    def call(self, x):
        # x is a complex128 tensor with shape (10, 9, 8)
        # Steps from the original code:
        # 1) Apply tf.raw_ops.Acos to x (input complex128 tensor)
        # 2) Create tensor = complex tensor from random normal distributions, shape (1, 10, 1, 1, 8), dtype complex128
        # 3) Apply tf.raw_ops.Xlogy(y=x, x=tensor) 
        # Return the result
        
        # Since `tensor` depends on a random normal, replicate the original behavior but using a fixed seed for determinism 
        # may be good practice but since original code uses random, we keep it.

        # Create the complex tensor as in the original snippet
        real_part = tf.random.normal([1, 10, 1, 1, 8], dtype=tf.float64)
        imag_part = tf.random.normal([1, 10, 1, 1, 8], dtype=tf.float64)
        tensor = tf.complex(real_part, imag_part)
        tensor = tf.cast(tensor, dtype=tf.complex128)

        # Apply Acos op on input x
        acos_res = tf.raw_ops.Acos(x=x)
        
        # Apply Xlogy op with y=acos_res and x=tensor
        result = tf.raw_ops.Xlogy(y=acos_res, x=tensor)

        return result

def my_model_function():
    # Instantiate MyModel
    return MyModel()

def GetInput():
    # Generate the complex128 input tensor expected by MyModel.call()
    # Based on the reproduction steps:
    # Shape: (10, 9, 8) with dtype complex128 created from random normal real and imag parts of float64
    real_part = tf.random.normal([10, 9, 8], dtype=tf.float64)
    imag_part = tf.random.normal([10, 9, 8], dtype=tf.float64)
    tensor = tf.complex(real_part, imag_part)
    tensor = tf.cast(tensor, dtype=tf.complex128)
    return tensor


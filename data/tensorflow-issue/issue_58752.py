# tf.random.stateless_uniform((2,), dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No extra layers needed since the model does direct TF operations

    @tf.function(jit_compile=True)
    def call(self, inputs=None):
        # The original code uses tf.random.stateless_uniform with shape [2], seed=(1,2),
        # minval=-5, maxval=5, dtype=tf.float64 and then computes:
        # cosh3 = cosh(Input_0)
        # cosh6 = cosh(cosh3)
        # tan11 = tan(cosh6)
        # and returns tan11.
        # The inputs argument is not used since input is generated inside call.
        input_0 = tf.random.stateless_uniform(
            shape=[2], seed=(1, 2),
            minval=-5., maxval=5., dtype=tf.float64)
        cosh3 = tf.math.cosh(input_0)
        cosh6 = tf.math.cosh(cosh3)
        tan11 = tf.math.tan(cosh6)
        return tan11

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The model's call does not take input, input is internally generated using a fixed seed.
    # To fulfill the interface, return None (or an empty tensor), as inputs argument is not used.
    return None


# tf.random.uniform((14, 2, 25, 53), dtype=tf.float32) ← inferred input shape from the issue code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Variable p1 shape matches the shape used in the original code: [2, 34, 35, 25]
        # dtype: float32
        self.p1 = tf.Variable(tf.random.uniform(shape=[2, 34, 35, 25], dtype=tf.float32))

    @tf.function(jit_compile=True)
    def call(self, inp):
        # Input tensor shape: [14, 2, 25, 53], dtype: float32
        # Perform atrous/dilated convolution with rate=2, padding VALID as in original code
        astconv = tf.nn.atrous_conv2d(self.p1, inp, rate=2, padding="VALID")
        # Round the convolution output
        round_ast = tf.round(astconv)

        # The issue revolves around inconsistent results between eager and XLA compiled modes on GPU
        # caused by tf.round after atrous_conv2d.
        # Since multiple outputs are returned in the original issue code, return both astconv and round_ast
        return astconv, round_ast

def my_model_function():
    # Instantiate and return the model with variable initialization
    return MyModel()

def GetInput():
    # Create a random tensor matching expected input shape [14, 2, 25, 53], dtype float32
    # Use uniform distribution like in original code for inputs
    return tf.random.uniform(shape=[14, 2, 25, 53], dtype=tf.float32)


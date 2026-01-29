# tf.random.normal((1, 1, 1), dtype=tf.float32) ← input shape and dtype inferred from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable layers are defined in the original code;
        # the main operation is tf.raw_ops.RandomGammaGrad with a generated alpha param.

    @tf.function(jit_compile=True)
    def call(self, x):
        # Generate the alpha parameter tensor as in the issue: shape [9,8,8,8,1,7,1], dtype float32
        alpha = tf.random.normal([9, 8, 8, 8, 1, 7, 1], dtype=tf.float32)
        # Apply RandomGammaGrad operator using the input sample `x` and generated alpha
        # The original code shows the sample input shape is [1,1,1] (likely scalar) but RandomGammaGrad expects sample and alpha shapes;
        # The exact broadcasting rules for RandomGammaGrad are complicated, but we replicate usage as is.
        result = tf.raw_ops.RandomGammaGrad(sample=x, alpha=alpha)
        return result

def my_model_function():
    # Return an initialized instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the sample input required by MyModel: shape [1,1,1], dtype float32
    # This matches the example in the issue and test code
    return tf.random.normal([1, 1, 1], dtype=tf.float32)


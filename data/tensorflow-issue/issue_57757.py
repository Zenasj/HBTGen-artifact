# tf.random.uniform((1,), dtype=tf.float32) ← Inferred input shape and dtype from example: shape=(1,), dtype=float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed; using raw ops as in example

    @tf.function(jit_compile=True)
    def call(self, x):
        # Following the original example:
        # y0 = softmax(x)
        y0 = tf.raw_ops.Softmax(logits=x)
        # y1 = sqrt(x)
        y1 = tf.sqrt(x)
        # y2 = pow(y0, y1)
        y2 = tf.pow(y0, y1)
        return y2

def my_model_function():
    return MyModel()

def GetInput():
    # The example used a 1-D tensor with negative single value [-0.3523314] dtype float32
    # This input triggers the interesting behavior regarding pow and NaN.
    return tf.constant([-0.3523314], dtype=tf.float32)


# tf.random.uniform((B, H, W, C), dtype=...) ← Input shape and dtype not specified in original issue;
# The code snippet shows a method with an int type annotation on an input scalar, so assume scalar input tensor of dtype tf.int32.
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No layers defined as the original snippet just had a stub method with a type annotation.

    @tf.function
    def call(self, x: tf.Tensor):
        # The original issue was about type annotations causing errors on save,
        # so here we provide a minimal pass-through model that accepts a tensor input.
        # The type annotation here uses tf.Tensor instead of int, based on later user comment.
        return x

def my_model_function():
    # Return an instance of MyModel; no weights or special initialization needed.
    return MyModel()

def GetInput():
    # Return a scalar int32 tensor input that matches the expected input in call().
    # The original example had an int argument; we convert that to a tensor of shape ().
    return tf.random.uniform((), minval=0, maxval=10, dtype=tf.int32)


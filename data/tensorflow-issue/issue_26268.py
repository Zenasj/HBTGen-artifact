# tf.random.uniform((), dtype=tf.float32)  <- The example input from the issue is a scalar tensor

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed, purely demonstrating control flow on tensor input

    def call(self, x):
        # The model's logic depends on conditional checking on tensor value.
        # Direct Python 'if' on tensor is not allowed during graph execution.
        # Use tf.cond to dynamically branch based on tensor value,
        # so it works both eagerly and graph-compiled.
        return tf.cond(
            x > 0,
            lambda: x + 1,
            lambda: x - 1
        )

def my_model_function():
    # Return an instance of MyModel
    # No special arguments needed, `dynamic=True` is not supported in recent TF.
    # run_eagerly can be set externally when compiling if eager execution is desired during training.
    return MyModel()

def GetInput():
    # Return a scalar tensor input compatible with MyModel call.
    # Use uniform random float scalar in range [-1, 1].
    return tf.random.uniform(shape=(), minval=-1, maxval=1, dtype=tf.float32)


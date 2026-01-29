# tf.random.uniform((B, 1), dtype=tf.float32)  # Input shape inferred from Input(shape=(1,)) in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers since the example layer simply returns inputs unchanged.
        # The main purpose is to reproduce the behavior that triggers the AutoGraph warning
        # when a multi-line string with backslash is used inside call().

    def call(self, inputs):
        # Following the original example in the issue's reproducible code,
        # deliberately use a multi-line string joined by backslash to illustrate the AutoGraph issue.
        s = "foo" \
            "bar"
        # Using tf.print instead of print to keep TF graph compatibility.
        tf.print(s)
        return inputs

def my_model_function():
    # Return an instance of MyModel with no special initializations
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape expected by MyModel's call/input shape (batch unknown, 1)
    # Using batch size 4 as an example
    return tf.random.uniform((4, 1), dtype=tf.float32)


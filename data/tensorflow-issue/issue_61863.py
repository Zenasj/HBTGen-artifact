# tf.random.uniform((4,), dtype=tf.float32) ← From input shape [4]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # To avoid the error "pred must not be a Python bool",
        # use a TensorFlow boolean tensor instead of Python bool.
        # Here we create a constant boolean tensor True with shape [].
        pred = tf.constant(True)

        # Use tf.cond with this tensor predicate to select output.
        # Both branches just return x unchanged, replicating original logic.
        y = tf.cond(pred, lambda: x, lambda: x)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Return a float32 tensor matching the input shape [4], as in the example.
    return tf.random.uniform(shape=(4,), dtype=tf.float32)


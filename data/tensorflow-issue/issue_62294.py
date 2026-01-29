# tf.random.uniform((4, 4), minval=-1000000, maxval=1000000, dtype=tf.int64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We move the random exponent generation outside of call to ensure reproducibility per inference,
        # avoiding different outputs due to different random 'y' each call.
        # Also use int64 to reduce overflow issues.
        self.exponent = tf.random.uniform([4, 1], minval=0, maxval=1000000, dtype=tf.int64)

    @tf.function(jit_compile=True)
    def call(self, x):
        # Compute pow(x, exponent) using tf.raw_ops.Pow with fixed exponent.
        # Using int64 dtype for both inputs for stability.
        output = tf.raw_ops.Pow(x=x, y=self.exponent)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor of shape (4, 4) with int64 dtype matching the expected input of MyModel.
    # The input dtype should match exponent dtype to avoid implicit casts.
    return tf.random.uniform(
        shape=(4, 4),
        minval=-1000000,
        maxval=1000000,
        dtype=tf.int64,
    )


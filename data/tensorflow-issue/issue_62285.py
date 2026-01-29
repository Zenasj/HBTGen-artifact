# tf.random.normal((8, 8, 8), dtype=tf.float32) ← Input shape inferred from the issue's example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We do not have learnable parameters; the model will apply tf.raw_ops.TruncateMod with a fixed random 'y'
        # To ensure stable output between calls, we generate and fix y during initialization.
        # This avoids the variability caused by tf.random.normal inside the call method.
        self.y_const = tf.random.normal([8, 8, 8], dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, x):
        # The core operation that was checked in the issue:
        # tf.raw_ops.TruncateMod(x=x, y=self.y_const) 
        # We fix y_const to avoid randomness and facilitate consistent outputs between jit and non-jit
        result = tf.raw_ops.TruncateMod(x=x, y=self.y_const)
        return result


def my_model_function():
    # Instantiate and return the MyModel instance.
    return MyModel()


def GetInput():
    # Return a random tensor of shape (8, 8, 8) with dtype float32 matching the model's input shape.
    return tf.random.normal([8, 8, 8], dtype=tf.float32)


# tf.random.uniform((1024, 1024, 1024), dtype=tf.float32) ← Input example for inferred large tensor context

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a large variable (4 GiB) similar to the referenced example.
        # We do this on first call, but to comply with TF2 and model saving behavior, 
        # we build it here eagerly as a Variable.
        # For practical reasons, we initialize it here once.
        # The large variable simulates the issue with large constants.
        self.v = tf.Variable(
            tf.random.uniform((1024, 1024, 1024), dtype=tf.float32),
            trainable=False,
            name="large_variable"
        )

    def call(self, x):
        # Multiply input by large variable, reduce sum as a scalar output.
        # Identity op for output naming consistent with signature example.
        out = tf.reduce_sum(x * self.v)
        return tf.identity(out, name="output_0")

def my_model_function():
    # Return an instance of the model
    # No weights are loaded externally here as we define variable internally.
    return MyModel()

def GetInput():
    # The model expects input shaped (1024, 1024, 1024) to multiply elementwise with self.v
    # However, computing a random tensor of shape (1024, 1024, 1024) floats (~4GB) here would be prohibitive.
    # The example code from the issue actually uses input shape (1024,) for the call,
    # but the variable shape is (1024,1024,1024), and multiplication is broadcasting.
    # So input shape used in multiplication must be broadcast compatible.
    #
    # The original example shows input of shape (1024,) and multiplies by v of shape (1024,1024,1024).
    # In TF broadcasting, (1024,) broadcast to (1024,1,1) then to (1024,1024,1024)
    #
    # We'll replicate this by generating input tensor shape (1024,) float32.
    return tf.random.uniform((1024,), dtype=tf.float32)


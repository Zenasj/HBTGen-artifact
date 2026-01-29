# tf.random.uniform((B, M, H, W, F), dtype=tf.float32), tf.random.uniform((B, M, O, H), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights, just demonstrating the einsum wrapped as a keras Lambda layer,
        # since direct tf.einsum layer serialization is problematic.
        # This captures the logic: out[b,m,o,w,f] = einsum of inputs x1, x2 with equation "bmhwf,bmoh->bmowf"
        self.einsum_layer = tf.keras.layers.Lambda(
            lambda inputs: tf.einsum('bmhwf,bmoh->bmowf', inputs[0], inputs[1]),
            name="einsum_layer"
        )

    def call(self, inputs):
        # inputs is a list or tuple of two tensors
        x1, x2 = inputs
        return self.einsum_layer([x1, x2])

def my_model_function():
    # Return an instance of MyModel with no additional parameters
    return MyModel()

def GetInput():
    # Based on the example:
    # x1 shape: (B, M, H, W, F) = (2, 2, 4, 4, 1)
    # x2 shape: (B, M, O, H) = (2, 2, 2, 4)
    # We'll use batch size 2 to match example, dtype float32
    x1 = tf.random.uniform((2, 2, 4, 4, 1), dtype=tf.float32)
    x2 = tf.random.uniform((2, 2, 2, 4), dtype=tf.float32)
    return (x1, x2)


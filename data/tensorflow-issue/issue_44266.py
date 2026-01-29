# tf.random.uniform((4, 1000, 1000, 10), dtype=tf.float32) ← inferred input shape from the issue NxHxWxC = 4x1000x1000x10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model was a no-op linear activation (just tf.keras.layers.Activation("linear"))
        # We replicate that behavior with a Lambda layer (identity)
        self.identity_layer = tf.keras.layers.Activation("linear")

    def call(self, inputs, training=False):
        # Forward pass: just pass through inputs unchanged, mimicking the original "linear" activation
        return self.identity_layer(inputs)

def my_model_function():
    # Return an instance of MyModel with no additional initialization
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the shape described in the issue: (N=4, H=1000, W=1000, C=10)
    # We'll use tf.random.uniform with dtype float32 to match the numpy arrays in the example
    N, H, W, C = 4, 1000, 1000, 10

    # Using uniform random data as a reasonable stand-in for the standard normal numpy arrays in the example
    return tf.random.uniform((N, H, W, C), dtype=tf.float32)


# tf.random.uniform((1, 10, 3), dtype=tf.float32) ← based on the example inputs shape in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Custom "MyLayer" logic incorporated here with fix:
        # We assume input shape last dim is integer, and convert shape dims with .value if needed.
        # Since TensorFlow 2.x generally uses int for shape dims, but to be safe,
        # convert to int in build to avoid errors with initializers like identity.
        self.kernel = None  # placeholder for weights

    def build(self, input_shape):
        # input_shape is a tf.TensorShape or tuple; last dim is channels/features
        dim = input_shape[-1]
        # Defensive conversion in case dim is a TensorShape.Dim object
        if not isinstance(dim, int):
            dim = int(dim)
        # Add the weight with identity initializer, fixing the shape as integers
        self.kernel = self.add_weight(
            name="kernel",
            shape=(dim, dim),
            initializer=tf.initializers.identity(),
            trainable=True,
        )

    def call(self, inputs):
        # inputs shape: (batch, seq_len, dim)
        # self.kernel shape: (dim, dim)
        # We need to multiply inputs by kernel matrix on last dimension.
        # inputs: (B, S, D), kernel: (D, D) → output: (B, S, D)
        return tf.matmul(inputs, self.kernel)

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape based on example in issue: batch=1, seq_len=10, features=3
    # dtype float32 as typical for model inputs
    shape = (1, 10, 3)
    return tf.random.uniform(shape, dtype=tf.float32)


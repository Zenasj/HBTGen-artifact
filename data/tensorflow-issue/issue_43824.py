# tf.random.uniform((B, 1), dtype=tf.float32) ← input shape inferred from example: Input shape (None, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple Dense layer as in the repro example from the issue
        # This layer acts like the 'layer2' in the provided example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass reproduction:
        # 1. Pass input through dense layer
        # 2. Slice output with [:, :-1] which triggers auto-naming of strided_slice op layer
        # Here we reproduce the slicing that caused the duplicate name issue in the bug report.
        x = self.dense(inputs)
        x_sliced = x[:, :-1]  # This slicing triggers naming auto-generation for a strided_slice layer
        return x_sliced

def my_model_function():
    # Return a fresh instance of MyModel as a keras Model would be recreated or loaded
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input shape expected by MyModel
    # The input shape needs to be compatible with Dense layer (which expects last dim size 1)
    # Batch size B is chosen as 2 arbitrarily here, can be any positive integer.
    batch_size = 2
    input_shape = (batch_size, 1)
    # Use float32 as default dtype for tf.keras inputs
    return tf.random.uniform(input_shape, dtype=tf.float32)


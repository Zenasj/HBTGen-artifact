# tf.random.uniform((B, 3, 4), dtype=tf.float32) ← inferred input shape from example: input shape=(3,4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the toy example from the issue:
        # Model input shape: (3,4)
        # First Dense layer with 10 units
        self.dense1 = tf.keras.layers.Dense(10)
        # Second Dense layer with 1 unit
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        # The key difference in the issue is about slicing using ellipsis vs normal slice.
        # The problematic version used: x[..., 0]
        # The correct version used: x[:, :, 0]
        #
        # We implement slicing without ellipsis as recommended because ellipsis causes TFLite shape issues.
        # This returns shape (batch, 3) if input is (batch, 3, 4).
        #
        # This model replicates the working case from the issue.
        return x[:, :, 0]  # slice on the last dimension without ellipsis

def my_model_function():
    return MyModel()

def GetInput():
    # The input shape is (batch_size, 3, 4) as per example (batch is flexible)
    # Provide batch size 1 here for simplicity
    batch_size = 1
    input_shape = (batch_size, 3, 4)
    return tf.random.uniform(input_shape, dtype=tf.float32)


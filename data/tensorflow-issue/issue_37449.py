# tf.random.uniform((B,), dtype=tf.float32) ← Inferred input shape for the simple regression input (1D batch vector)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the trivial Dense model from the issue:
        # Single Dense layer with 1 output and input shape of [1]
        self.dense = tf.keras.layers.Dense(units=1, input_shape=(1,))

    def call(self, inputs):
        # Forward pass applies the dense layer to the input
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel, initialized but not loaded with any custom weights
    return MyModel()

def GetInput():
    # Generate a batch of 6 float inputs matching the example input xs from the code sample
    # Shape: (6, 1), dtype=tf.float32
    # Values are as in original example: [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    import numpy as np
    input_array = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_array = input_array.reshape((-1, 1))  # shape (6, 1)
    return tf.convert_to_tensor(input_array)


# tf.random.uniform((B, H, W, 512), dtype=tf.float32) ← Input is a 3D tensor representing batches of sequences of length H with feature size 512

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a 5-layer dense network each mapping 512 -> 1024 units
        # The example code used tf.keras.Sequential with 5 Dense(1024) layers
        self.dense_layers = [
            tf.keras.layers.Dense(1024, activation=None) for _ in range(5)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # To mirror the scenario with dynamic batch and sequence length,
    # we generate a random batch size and sequence length such that
    # batch_size * length ≈ 4096 (similar total tokens),
    # and feature size is fixed at 512.
    import random
    length = random.randint(1, 100)
    batch_size = max(1, int(4096 / length))
    # Generate a random tensor uniform between 0 and 1 for float32 dtype
    return tf.random.uniform((batch_size, length, 512), dtype=tf.float32)


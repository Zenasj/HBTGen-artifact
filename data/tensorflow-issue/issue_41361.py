# tf.random.uniform((B, 3), dtype=tf.float32) ← inferred input shape: batch size x 3 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For simplicity, the model just returns the input as output, matching the reported toy example
        # from the issue where input = output in the minimal repro.
        # In a real use, this could be any model.
        self.identity = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs):
        return self.identity(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random input tensor matching the input shape described above:
    # Batch size = 2 for example, features = 3, dtype float32 (common for keras models).
    # The original example shows inputs like [[0.1, 0.6, 0.9]], here we generalize with uniform floats.
    return tf.random.uniform((2, 3), dtype=tf.float32)


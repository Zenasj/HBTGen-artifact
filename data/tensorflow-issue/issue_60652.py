# tf.random.uniform((B, 1), dtype=tf.float32) ← Based on the provided model input shape Input(1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the simple Dense model as described in the issue:
        # Input shape: (1,), Dense(10) layer
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel

    # Note: The issue centers around the AdamW optimizer usage on Mac M1 with TF 2.12.
    # The provided model is simple: Input(1) -> Dense(10).
    # Also, no weights are specified, so default initialization is used.
    
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape (batch_size, 1)
    # Assumption: batch size can be flexible; choose 4 for example.
    # dtype float32 as typical for Keras models.
    batch_size = 4
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


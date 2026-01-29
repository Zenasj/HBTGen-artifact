# tf.random.uniform((B, 2), dtype=tf.float64) ← Input shape inferred from the generator output ([2], scalar label), batch size B dynamic

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple modeled version of the original Sequential model:
        # Input layer with size 2, Dense(1)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass through dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of the simple model.
    # In the original issue, model compilation uses Adam optimizer and MSE loss,
    # but compilation details don't belong in the model class directly.
    # This function returns the prepared MyModel instance.
    return MyModel()

def GetInput():
    # Returns a batch of inputs matching the shape expected by the model,
    # dtype=tf.float64 as per the generator output types.
    # Since batch size is dynamic, generate a small batch for testing.
    batch_size = 4
    # Generate random input of shape (batch_size, 2)
    return tf.random.uniform((batch_size, 2), dtype=tf.float64)


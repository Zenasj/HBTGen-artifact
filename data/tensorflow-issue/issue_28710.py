# tf.random.uniform((batch_size, 4), dtype=tf.float32) ← Input with shape (batch_size, 4) as in the example data

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple single dense layer with sigmoid activation, mirroring the given example
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of the simple model
    return MyModel()

def GetInput():
    # Generate a random input tensor matching shape (batch_size, 4)
    # Use batch_size=8 as a reasonable example size for testing
    batch_size = 8
    # Inputs are float32 with 4 features, matching the original example data shape
    return tf.random.uniform((batch_size, 4), dtype=tf.float32)


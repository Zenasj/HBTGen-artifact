# tf.random.uniform((B, 20, 20), dtype=tf.float32) ← Assumed input shape based on random_crop((20,20)) in the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model adapted from the issue example
        # Input shape (20,20) is flattened by Dense layer implicitly treating last axis as features
        # Since input is 2D, Dense layer applies on last axis; here we add a Flatten to be explicit
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel as in the example
    return MyModel()

def GetInput():
    # Generate a random tensor matching the input shape expected
    # The input shape is (20, 20) as per random_crop output in the issue.
    # Batch dimension is not included here since model expects single samples.
    # We provide a float32 tensor filled with uniform random values between 0 and 1.
    return tf.random.uniform((20, 20), dtype=tf.float32)


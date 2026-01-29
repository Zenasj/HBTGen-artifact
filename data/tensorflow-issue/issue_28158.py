# tf.random.uniform((B, 1), dtype=tf.float32) ← assuming input is a batch of vectors with shape (batch_size, 1) as in the provided example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Dense layer with 1 output unit and ReLU activation as per original example
        self.dense = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs):
        # Directly call the Dense layer, which will lazily build on first call
        return self.dense(inputs)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate a batch of input data shaped (batch_size, 1)
    # Here we choose batch size = 1 to mirror original example
    return tf.random.uniform((1, 1), dtype=tf.float32)


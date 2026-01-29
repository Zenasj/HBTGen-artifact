# tf.random.uniform((B, 10), dtype=tf.float32) for each input with batch size B and feature size 10
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two inputs, processed individually then concatenated
        self.dense_concat = tf.keras.layers.Concatenate()
        self.dense_output = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=False):
        # inputs is expected to be a tuple of two tensors
        inputA, inputB = inputs
        x = self.dense_concat([inputA, inputB])
        return self.dense_output(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random batch of data that matches the expected input format:
    # A tuple with two tensors of shape (batch_size, 10) each, dtype float32.
    batch_size = 32  # typical batch size for training
    inputA = tf.random.uniform((batch_size, 10), dtype=tf.float32)
    inputB = tf.random.uniform((batch_size, 10), dtype=tf.float32)
    return (inputA, inputB)


# tf.random.uniform((B, 2), dtype=tf.string) ← Input shape inferred from model Input(shape=(2,), dtype=tf.string)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize a lookup table mapping strings ["a", "b"] to int64 numbers [1, 2]
        names = tf.constant(["a", "b"])
        numbers = tf.constant([1, 2], dtype=tf.int64)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(names, numbers),
            default_value=-1,
        )
        
    def call(self, inputs):
        # Flatten inputs to 1D tensor of strings and lookup indices in the table
        flat_inputs = tf.reshape(inputs, [-1])
        return self.table.lookup(flat_inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input consistent with MyModel's input: shape (batch_size, 2) strings "a" or "b"
    # To simulate typical input, produce batch size 3 with random choice of "a" and "b"
    batch_size = 3
    import numpy as np
    # Generate random choices from ["a", "b"] of shape (batch_size, 2)
    choices = np.random.choice(["a", "b"], size=(batch_size, 2))
    return tf.constant(choices, dtype=tf.string)


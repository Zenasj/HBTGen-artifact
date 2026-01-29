# tf.random.uniform((B, 5), dtype=tf.float32), tf.random.uniform((B, 10), dtype=tf.float32) ← input shapes for inputs x and y respectively

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(MyModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [tf.keras.layers.Dense(u) for u in self.hidden_units]

    def call(self, inputs):
        # Expecting inputs as a list or tuple of two tensors (x, y)
        x, y = inputs  # Unpack inputs
        # Concatenate along last dimension (axis=1)
        combined = tf.concat((x, y), axis=1)
        for layer in self.dense_layers:
            combined = layer(combined)
        return {'x': combined}

def my_model_function():
    # Instantiate MyModel with example hidden units same as original example
    return MyModel([16, 16, 10])

def GetInput():
    # Return a tuple of two random tensors, shapes aligned with original example: (B, 5) and (B, 10)
    # Batch size chosen as 1 to mimic example, can be changed as needed
    x = tf.random.uniform((1, 5), dtype=tf.float32)
    y = tf.random.uniform((1, 10), dtype=tf.float32)
    return (x, y)


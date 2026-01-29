# tf.random.uniform((B, None), dtype=tf.float32)  # Input shape: batch size B, variable sequence length

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The example from the issue sums inputs along axis=1 (variable length dimension)
        # then applies a Dense layer with 1 unit.
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # inputs: shape (B, variable_length)
        # Compute sum along axis=1 with keepdims to keep shape compatible with Dense
        summed = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
        output = self.dense(summed)
        return output

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Generate a random tensor input with shape (batch_size, seq_len)
    # Batch size is 4, seq_len is variable - choose e.g. 7 as an example
    # Because the model sums across axis=1, input must be at least 2D.
    batch_size = 4
    seq_len = 7  # arbitrary variable-length sequence size for testing
    # Using float32 input as typical Keras default dtype
    return tf.random.uniform((batch_size, seq_len), dtype=tf.float32)


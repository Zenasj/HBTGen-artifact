# tf.random.uniform((batch_size, window_size), dtype=tf.int64) <- Input shape inferred from example sliding windows and batching

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model expects input shape (batch_size, window_size=10)
        # and produces output shape (batch_size, 5)
        self.dense = tf.keras.layers.Dense(5)

    def call(self, inputs, training=False):
        # inputs shape: [batch_size, 10]
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Based on the example in the issue:
    # The sliding window size is 10 for inputs, batch size 20.
    # Let's generate a random tensor matching shape (20,10).
    batch_size = 20
    window_size = 10
    # Using dtype float32 since model Dense expects floats
    return tf.random.uniform((batch_size, window_size), dtype=tf.float32)


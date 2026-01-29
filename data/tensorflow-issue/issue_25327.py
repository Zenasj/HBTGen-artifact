# tf.random.uniform((B,)) with dtype=tf.int32 ← Input shape: (batch_size,), dtype int32 tokens (indices) for embedding

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the example, input_dim=100, output_dim=16, input_length=1
        # We flatten output after embedding to keep consistent output shape
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=16, input_length=1, name="state")

    def call(self, inputs):
        # inputs shape: (batch_size, 1)
        x = self.embedding(inputs)  # shape: (batch_size, 1, 16)
        x = tf.squeeze(x, axis=1)   # shape: (batch_size, 16) - flatten sequence dim since input_length=1
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random integer tensor of shape (B, 1) with values in [0, 100),
    # dtype int32, matching the Embedding input requirements
    batch_size = 4  # arbitrary batch size for example
    return tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=100, dtype=tf.int32)


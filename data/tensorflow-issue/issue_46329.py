# tf.random.uniform((32, 10), dtype=tf.int32) ← Input is a batch of sequences of token IDs, batch size=32, sequence length=10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create an embedding layer with fixed vocab size=1000, embedding dim=64, input_length=10
        # trainable=False to reflect the original intention
        self.embedding = tf.keras.layers.Embedding(
            input_dim=1000,
            output_dim=64,
            input_length=10,
            trainable=False  # explicitly set to non-trainable as per issue
        )

    def call(self, inputs):
        # inputs expected as integer token IDs with shape (batch_size, 10)
        return self.embedding(inputs)

def my_model_function():
    # Instantiate MyModel with embedding layer non-trainable
    return MyModel()

def GetInput():
    # Return a random int tensor of shape (32, 10), values in [0, 999]
    # matching input expected by the embedding layer
    return tf.random.uniform((32, 10), minval=0, maxval=1000, dtype=tf.int32)


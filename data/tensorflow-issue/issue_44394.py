# tf.random.uniform((B,), dtype=tf.int32) ← Input is a batch of integer indices representing categorical lookup keys

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: 10 rows, each mapped to a 12-dimensional vector.
        # Initialized to zeros for reproducibility with original example.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=10,
            output_dim=12,
            embeddings_initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs):
        # inputs shape: (batch_size, ) - integer indices for embedding lookup
        return self.embedding(inputs)


def my_model_function():
    # Instantiate and return the model instance
    return MyModel()


def GetInput():
    # Create a batch of indices as input to the embedding layer.
    # Using batch size 4 here for demonstration; values in [0, 9] inclusive.
    batch_size = 4
    inputs = tf.random.uniform(shape=(batch_size,), minval=0, maxval=10, dtype=tf.int32)
    return inputs


# tf.random.uniform((B, 1), maxval=40000000, dtype=tf.int64) ← assumed input shape and dtype

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=40000000, embedding_dim=10):
        super().__init__()
        # Use int64 input dtype to avoid collisions as per issue discussion.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            dtype=tf.float32,  # embeddings are float32
            embeddings_initializer='uniform',
            input_length=1
        )
    
    def call(self, inputs):
        # inputs expected as (batch_size, 1) int64 tensor of indices
        return self.embedding(inputs)

def my_model_function():
    # Instantiate model with large vocabulary size and embedding dim 10
    return MyModel()

def GetInput():
    # Generate random batch of integer indices within vocab size (up to 40 million)
    # Batch size chosen arbitrarily as 4 for demonstration (can be changed)
    batch_size = 4
    vocab_size = 40000000
    # Input shape is (batch_size, 1) to match embedding input requirement
    return tf.random.uniform(
        shape=(batch_size, 1),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int64
    )


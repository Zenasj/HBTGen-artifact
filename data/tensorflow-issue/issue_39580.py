# tf.ragged.constant([[...]], dtype=tf.int32) for inputs 'a' and 'b', shape=(B, None)

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embeddings: vocab size 10, embedding dim 3
        self.embedding_a = Embedding(10, 3)
        self.embedding_b = Embedding(10, 3)
        # Dense layer to produce final scalar output
        self.dense = Dense(1)
        # Concatenate layer - can use tf.concat in call, but Keras layer gives easy usage

    def call(self, inputs):
        # Inputs expected: tuple or list of two ragged tensors of shape (batch, None)
        a, b = inputs
        
        # Embed inputs
        embed_a = self.embedding_a(a)  # Ragged tensor output: (batch, ragged_len, 3)
        embed_b = self.embedding_b(b)
        
        # Reduce sum over axis=1 (the ragged dimension)
        # Use tf.reduce_sum instead of Lambda with direct function reference,
        # to avoid serialization issues with Lambda layers and ragged tensors.
        embed_a_reduced = tf.reduce_sum(embed_a, axis=1)  # shape (batch, 3)
        embed_b_reduced = tf.reduce_sum(embed_b, axis=1)  # shape (batch, 3)

        # Concatenate embeddings on last dim
        concat = tf.concat([embed_a_reduced, embed_b_reduced], axis=1)  # shape (batch, 6)

        # Dense layer to scalar
        out = self.dense(concat)  # shape (batch, 1)

        return out

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Create ragged inputs matching the expected input signature:
    # Two ragged int32 tensors with shape (batch, None), for example batch=3
    a_input = tf.ragged.constant([[0], [], [1]], dtype=tf.int32)
    b_input = tf.ragged.constant([[0], [], [1]], dtype=tf.int32)
    return (a_input, b_input)


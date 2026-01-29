# tf.random.uniform((B, 3), dtype=tf.int64) ← Input is integer indices tensor of shape (batch_size, 3)

import tensorflow as tf
import tensorflow.keras as keras


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assume the vocabulary size and embedding dims based on the example
        # The vocab corresponds to ['Elephant', 'I am a sentence...', 'Universal Sentence Encoder ...']
        # Plus a special OOV or mask tokens as usual in StringLookup.
        vocab_size = 3 + 1  # inferred: 3 messages + 1 for OOV or mask token
        embedding_dim = 20

        # Embedding layer as in example
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=False,  # Not explicitly mentioned, assume False
            name="embedding_layer"
        )
        # Dense layer outputting a single unit with sigmoid activation for binary output
        # The example tries to do categorical but only uses 1 unit with sigmoid.
        # For demonstration, output 3 units with softmax to align with 3-class scenario.
        self.dense = keras.layers.Dense(units=3, activation="softmax", name="output_dense")

    def call(self, inputs, training=False):
        """
        inputs: tf.int64 tensor of shape (batch_size, 3)
        Each row corresponds to string token ids after lookup.
        """
        # inputs shape: (batch_size, 3)
        embedded = self.embedding(inputs)  # shape: (batch_size, 3, embedding_dim)
        # Reduce mean over sequence dimension (axis=1)
        pooled = tf.reduce_mean(embedded, axis=1)  # (batch_size, embedding_dim)

        # Pass through final dense layer. Output shape: (batch_size, 3)
        output = self.dense(pooled)
        return output


def my_model_function():
    # Return an instance of the model
    # This model is untrained and weights randomly initialized
    return MyModel()


def GetInput():
    # Input is a tf.int64 tensor of shape (batch_size, 3),
    # with integer ids matching a vocabulary of size 4 (3 known words + 1 OOV)
    # We'll generate random indices in [0, 3] inclusive.
    batch_size = 5  # arbitrary batch size
    input_tensor = tf.random.uniform(
        shape=(batch_size, 3),
        minval=0,
        maxval=4,
        dtype=tf.int64
    )
    return input_tensor


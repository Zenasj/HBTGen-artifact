# tf.random.uniform((B,), dtype=tf.string) ← input is a batch of strings (shape [batch_size]) as expected by the hub.KerasLayer embedding

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, hub_layer=None):
        super().__init__()
        # If no external hub layer provided, create a placeholder hub layer for demonstration
        # In practice pass a loaded hub.KerasLayer (e.g. from TF Hub text embedding model)
        if hub_layer is None:
            # Here we create a dummy embedding layer that accepts strings and returns a fixed-size embedding
            # This is a placeholder to make the model runnable without requiring internet or TF Hub.
            # It uses a StringLookup + Embedding as a minimal stub.
            vocab = ["this", "is", "a", "dummy", "embedding"]
            self.string_lookup = layers.StringLookup(vocabulary=vocab, output_mode='int', oov_token="[OOV]")
            self.embedding_layer = layers.Embedding(input_dim=len(vocab)+2, output_dim=16)
            self.embedding = self._dummy_embedding  # use an internal fn to embed
        else:
            self.embedding = hub_layer  # Expected to accept (batch of strings) with shape (batch_size,)

        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def _dummy_embedding(self, x):
        # x is tf.string tensor shape (batch_size,)
        # tokenize into tokens, map to int ids, embed and average to simulate embedding
        # Note: this is a simplified stand-in, do not use in production
        tokens = tf.strings.split(x)
        token_ids = self.string_lookup(tokens)
        embedded = self.embedding_layer(token_ids)
        # average embeddings per example to collapse sequence dimension
        embedded = tf.reduce_mean(embedded, axis=1)
        return embedded

    def call(self, x):
        # x has shape (batch_size,), dtype string
        x = self.embedding(x)  # shape (batch_size, embed_dim)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # In practice, you should load a real hub.KerasLayer like below:
    # hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2", input_shape=[], dtype=tf.string, trainable=True)
    # return MyModel(hub_layer)
    #
    # Here we return MyModel with dummy embedding for self-contained example
    return MyModel()

def GetInput():
    # Return a batch of strings (tf.Tensor of shape (batch_size,))
    # This matches the input signature expected by hub layers and the model call method assuming string inputs.
    batch_size = 4
    # For demonstration, random dummy strings composed of words from vocab or fixed set
    sample_sentences = [
        "this is good",
        "dummy embedding test",
        "tensorflow keras hub",
        "model class example"
    ]
    # Convert list to tf constant tensor of shape (batch_size,)
    return tf.constant(sample_sentences[:batch_size])


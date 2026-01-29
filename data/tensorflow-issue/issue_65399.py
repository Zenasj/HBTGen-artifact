# tf.random.uniform((B, None), dtype=tf.int64) ← Input is a ragged 1D sequence of token ids (batch of variable-length sequences)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        hash_buckets = 1000
        embedding_dim = 16
        lstm_units = 32
        dense_units = 32

        # Since tf.keras.Input and InputLayer no longer accept ragged=True as argument in TF2.16+,
        # our model will accept standard tf.RaggedTensor input directly in call()
        self.embedding = tf.keras.layers.Embedding(hash_buckets, embedding_dim)
        # LSTM without bias as in example
        self.lstm = tf.keras.layers.LSTM(lstm_units, use_bias=False)
        self.dense1 = tf.keras.layers.Dense(dense_units)
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs is expected to be a tf.RaggedTensor of shape [batch, None] with dtype tf.int64
        # Convert ragged tensor to padded tensor, masking is handled automatically in Embedding+LSTM if ragged.
        x = self.embedding(inputs)
        # The LSTM layer supports ragged inputs starting TF 2.5+, so we can pass ragged tensor to LSTM directly
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Just return an instance of MyModel; no pre-loaded weights assumed
    return MyModel()

def GetInput():
    # Produce a ragged tensor input mimicking the example:
    # Batch of 4 sentences, variable number of tokens per sentence
    # Each value is an int64 token id within [0, 999]
    # We'll use tf.ragged.constant to create a batch of sequences like split hashed words
    import numpy as np

    # Example tokenized sequences as ragged lists of int IDs (hash bucket values)
    data = [
        [5, 120, 17, 95, 200],      # e.g. "What makes you think she"
        [100, 30, 200],             # e.g. "She turned me"
        [201, 300],                 # e.g. "A newt"
        [90, 40, 150]               # e.g. "Well, I got"
    ]
    ragged_input = tf.ragged.constant(data, dtype=tf.int64)
    return ragged_input


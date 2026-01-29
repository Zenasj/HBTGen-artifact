# tf.random.uniform((B,)) ← Input shape is a 1D integer sequence (token indices) of length maxlen (unknown here)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, maxlen=100, vocab_size=10000):
        super().__init__()
        # Assumptions:
        # - maxlen and vocab_size are placeholders; user can adjust.
        # - Embedding output dim = 128 as per original code.
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=128, trainable=True)
        self.masking = layers.Masking(mask_value=0, name='mask')
        # Bidirectional GRU with 128 units, not returning sequences, as in original code
        self.bi_gru = layers.Bidirectional(layers.GRU(128, return_sequences=False))
        self.dense = layers.Dense(1, activation='sigmoid', name='mlp2')
    
    def call(self, inputs):
        # inputs: shape (batch_size, maxlen), int32 tokens
        x = self.embedding(inputs)           # (B, maxlen, 128)
        x = self.masking(x)                  # (B, maxlen, 128), mask zeros
        x = self.bi_gru(x)                   # (B, 256) because bidirectional concat
        outputs = self.dense(x)              # (B, 1), sigmoid output
        return outputs


def my_model_function():
    # Return an instance with default maxlen and vocab_size
    return MyModel()

def GetInput():
    # Return a random int tensor input with shape (batch_size, maxlen) 
    # For batching compatibility, set B=4 as example, maxlen=100 (same as above),
    # vocab_size=10000 - token indices in [0, vocab_size-1]
    # Input dtype is int32 for embedding
    batch_size = 4
    maxlen = 100
    vocab_size = 10000
    return tf.random.uniform(shape=(batch_size, maxlen), minval=0, maxval=vocab_size, dtype=tf.int32)


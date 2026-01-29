# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape is [batch_size, variable_seq_len], integer token IDs for embedding input.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self,
                 vocab_size=5000,
                 embedding_dim=256,
                 rnn_units=1024,
                 batch_size=64):
        super().__init__()
        # Embedding layer with batch_input_shape: batch_size and variable time dimension (None).
        # Using batch_input_shape to support stateful RNN.
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            batch_input_shape=[batch_size, None]
        )
        # Stateful GRU layer, returning sequences
        self.gru = layers.GRU(
            units=rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'
        )
        # Dense output layer projecting to vocab size logits
        self.dense = layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs shape: (batch_size, seq_len) integer token IDs
        returns: logits shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(inputs)
        x = self.gru(x)
        logits = self.dense(x)
        return logits

def my_model_function():
    # Instantiate MyModel with some default sizes
    # Default batch_size=64, vocab_size=5000, embedding_dim=256, rnn_units=1024
    return MyModel()

def GetInput():
    # Create a sample valid input matching MyModel expectations:
    # Tensor shape: [batch_size, seq_len] with integer tokens in [0, vocab_size).
    batch_size = 64  # must match the model's batch input shape
    seq_len = 10     # example sequence length
    vocab_size = 5000
    # Generate random integer token IDs between 0 and vocab_size-1
    return tf.random.uniform(
        shape=(batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
    )


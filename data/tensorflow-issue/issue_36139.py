# tf.random.uniform((batch_size, seq_len), dtype=tf.int32) ← Input is a batch of integer sequences representing token IDs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, seq_len=5, vocab_size=10, hidden_size=20):
        super().__init__()
        # Embedding with mask_zero=True to handle padding tokens (0)
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)
        # GRU layer - this will use CuDNN kernel if possible (right-padded sequences)
        self.gru = tf.keras.layers.GRU(hidden_size)
        
    def call(self, inputs, training=False):
        """
        inputs: integer sequences of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, hidden_size) - GRU outputs
        """
        # embeddings output shape: (batch_size, seq_len, hidden_size)
        x = self.embedding(inputs)
        # Pass through GRU; the embedding layer provides mask automatically because mask_zero=True
        out = self.gru(x)
        return out

def my_model_function():
    # Return an instance of MyModel with default parameters matching the issue example
    return MyModel()

def GetInput():
    # Produce a random batch of integer sequences with padding (0) allowed
    # We produce batch_size=2, seq_len=5 to match the original example
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    
    # Generate two sequences:
    # First sequence: random tokens from [1, vocab_size-1], ensuring some non-zero tokens
    # Second sequence: all zeros to simulate fully padded sequence
    seq1 = tf.random.uniform((seq_len,), minval=1, maxval=vocab_size, dtype=tf.int32)
    seq2 = tf.zeros((seq_len,), dtype=tf.int32)
    
    # Stack to form batch
    batch = tf.stack([seq1, seq2], axis=0)
    return batch


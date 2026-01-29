# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape: batch size B, variable sequence length, integer tokens (text encoded)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=16, lstm_units=32):
        super().__init__()
        # Embedding with mask_zero=True to handle variable-length padded sequences
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True)
        # Bidirectional LSTM as in original model - uses cuDNN when on GPU if available
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units))
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.bi_lstm(x, training=training)
        x = self.classifier(x)
        return x

def my_model_function():
    # The imdb_reviews/subwords8k dataset from TFDS uses a vocabulary size of 8185 in the tutorial,
    # but since we are not loading dataset here, choose a placeholder vocab size consistent with official subwords8k.
    # (In practice this number should come from the encoder.)
    vocab_size = 8185
    embedding_dim = 16
    lstm_units = 32
    return MyModel(vocab_size, embedding_dim, lstm_units)

def GetInput():
    # Return a random batch of integer sequences representing tokenized text
    # Assumptions based on imdb_reviews/subwords8k dataset:
    # - Batch size: 10 (small batch to avoid OOM issues shown in the issue)
    # - Sequence length: variable, pad with zeros. We'll generate sequences of len 50 here.
    # - Integer values between 1 and vocab_size-1, 0 reserved for padding (mask_zero=True)
    batch_size = 10
    seq_len = 50
    vocab_size = 8185

    # Generate random int32 tensor with values in [1, vocab_size-1] (0 reserved for padding)
    # Here we do not pad with zeros explicitly since all are > 0 (mask_zero=True handles masking)
    input_data = tf.random.uniform(
        shape=(batch_size, seq_len), minval=1, maxval=vocab_size, dtype=tf.int32)
    return input_data


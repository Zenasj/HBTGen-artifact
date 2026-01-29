# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape is (batch_size, sequence_length), variable sequence length (text sequences)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 16
        # Vocabulary size inferred from typical imdb_reviews/subwords8k dataset (default ~8000)
        # Use mask_zero=False due to known issues with masking on cuDNN LSTM bidirectional in TF2.0
        self.vocab_size = 8192  # approximate for example; can be adjusted as needed

        self.embedding = layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=False)
        # Bidirectional LSTM with 32 units, single layer, default activation='tanh' to match tutorial
        self.bidir_lstm = layers.Bidirectional(layers.LSTM(32))
        # Final sigmoid for binary classification
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: tf.Tensor[int32], shape (batch_size, seq_length)
        """
        x = self.embedding(inputs)  # (B, seq_len, embedding_dim)
        # NOTE: Avoid mask_zero=True embedding + cudnn LSTM bidir due to known TF2.0 bug.
        # Masking omitted for compatibility with cuDNN
        x = self.bidir_lstm(x)  # (B, 64)
        x = self.dense(x)  # (B, 1)
        return x

def my_model_function():
    """
    Returns an instance of MyModel suitable for binary text classification with bidirectional LSTM.
    """
    model = MyModel()
    # Typical compilation as in the tensorflow tutorial
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    """
    Returns a random tensor input matching the expected input of MyModel.
    We simulate a batch of tokenized sequences with variable length.

    Due to cuDNN issues on Windows described in the issue,
    here the input shape is fixed length but no mask_zero usage to avoid known bugs.

    Returns:
      input_tensor: tf.Tensor[int32], shape (batch_size, seq_length)
    """
    batch_size = 10
    seq_length = 50  # fixed seq length for testing; variable length sequences possible but may cause cudnn issue

    # Random ints in vocab range [0, vocab_size)
    input_tensor = tf.random.uniform(
        (batch_size, seq_length),
        minval=0,
        maxval=8192,
        dtype=tf.int32
    )
    return input_tensor


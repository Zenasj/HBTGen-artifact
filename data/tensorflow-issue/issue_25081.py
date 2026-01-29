# tf.random.uniform((batch_size, sequence_length), dtype=tf.int32) ← inferred input shape: input token ids for text generation model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=65, embedding_dim=256, rnn_units=1024, batch_size=64):
        """
        A fused model encapsulating:
        - Embedding layer
        - GRU or CuDNNGRU layer depending on device
        - Dense output layer

        This is reconstructed based on the text_generation tutorial
        snippet discussed in the issue.

        Args:
          vocab_size (int): size of the vocabulary
          embedding_dim (int): dimension of embedding vectors
          rnn_units (int): number of RNN units
          batch_size (int): batch size for input sequences
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        # Use CuDNNGRU if GPU is available, else vanilla GRU.
        # This replicates the issue cause where CuDNNGRU and GRU
        # variables are incompatible between GPU and CPU.
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                self.rnn_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        else:
            self.gru = tf.keras.layers.GRU(
                self.rnn_units,
                return_sequences=True,
                return_state=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform'
            )

        self.fc = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden):
        """
        Forward pass.

        Args:
          x: input tensor of shape (batch_size, seq_length) with integer token IDs
          hidden: tensor of shape (batch_size, rnn_units) representing RNN state

        Returns:
          output logits of shape (batch_size, seq_length, vocab_size)
          and new hidden state of shape (batch_size, rnn_units)
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        x = self.fc(output)
        return x, state

    def initialize_hidden_state(self):
        """
        Returns a zero tensor as the initial hidden state.
        """
        return tf.zeros((self.batch_size, self.rnn_units))

def my_model_function():
    """
    Returns an instance of MyModel with default parameters.

    These parameters are based on the text_generation tutorial.
    """
    return MyModel()

def GetInput():
    """
    Returns a random integer tensor simulating input token IDs.

    Input shape: (batch_size, seq_length)

    We use batch_size and sequence length from the model defaults:
    batch_size=64, seq_length=100 (typical length for text generation).

    Vocabulary size is 65 by default in model.
    """
    batch_size = 64
    seq_length = 100
    vocab_size = 65
    return tf.random.uniform(
        (batch_size, seq_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )


# tf.random.uniform((B, None, 100), dtype=tf.float32) ← Inferred input shape: batch_size x sequence_length x input_dim=100, sequence length is variable (None)
import tensorflow as tf
import numpy as np


class MyModel(tf.keras.Model):
    """
    An autoencoder model composed of two Bidirectional LSTM layers (encoder and decoder),
    supporting variable-length sequences via input lengths and sequence_mask.
    """

    def __init__(self, input_dim=100, embed_dim=50):
        super(MyModel, self).__init__()
        # Encoder: Bidirectional LSTM with sum merge mode
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embed_dim, return_sequences=True),
            merge_mode='sum', name='encoder')
        # Decoder: Bidirectional LSTM with sum merge mode
        self.decoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(input_dim, return_sequences=True),
            merge_mode='sum', name='decoder')

    def call(self, inputs):
        """
        Expected input: tuple/list of (inputs, sizes)
          - inputs: float32 tensor [batch_size, max_seq_len, input_dim]
          - sizes: int32 tensor [batch_size] representing actual sequence lengths

        Applies masking internally using tf.sequence_mask.
        """
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError("Input to MyModel must be a tuple/list: (inputs, sizes)")

        sequences, sizes = inputs
        mask = tf.sequence_mask(sizes, maxlen=tf.shape(sequences)[1])
        x = self.encoder(sequences, mask=mask)
        x = self.decoder(x, mask=mask)
        return x


def my_model_function():
    """
    Returns an instance of MyModel with default input_dim=100 and embed_dim=50.
    """
    return MyModel(input_dim=100, embed_dim=50)


def GetInput():
    """
    Returns a valid random input tuple (inputs, sizes) matching MyModel's expected input:
      - inputs: float32 tensor [batch_size, max_seq_len, input_dim=100]
      - sizes: int32 vector [batch_size] with sequence lengths <= max_seq_len

    Here:
      - batch_size = 64
      - max_seq_len = 500
      - input_dim = 100
    """
    batch_size = 64
    max_seq_len = 500
    input_dim = 100

    # Randomly choose actual sequence lengths for each batch element
    np.random.seed(0)
    sizes_np = np.random.choice(max_seq_len, size=batch_size)
    # Create zero-padded inputs of shape (batch_size, max_seq_len, input_dim)
    inputs_np = np.random.normal(size=(batch_size, max_seq_len, input_dim)).astype(np.float32)
    # Zero out padding positions beyond actual length
    for i, length in enumerate(sizes_np):
        if length < max_seq_len:
            inputs_np[i, length:] = 0.0

    inputs_tf = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
    sizes_tf = tf.convert_to_tensor(sizes_np, dtype=tf.int32)
    return (inputs_tf, sizes_tf)


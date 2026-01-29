# tf.random.uniform((64, None), dtype=tf.int32) ← Input shape is a batch of sequences with variable length, batch size 64, integer tokens

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # From the provided code snippet:
        # Embedding layer maps vocabulary size to embedding dimension 64
        # Bidirectional LSTM with 64 units
        # Dense layer with 64 units and relu activation
        # Final Dense layer outputting a scalar (for binary classification logits)
        #
        # Note: Since the vocabulary size is dataset-dependent, we will set a placeholder
        # vocab_size here and expect it as an argument when creating the model.
        #
        # The original code uses subwords8k encoder, so vocab size = 8192 approx.
        # We'll default to 8192 but allow override.
        
        # These layers mimic the original model structure with Bidirectional LSTM
        self.embedding = tf.keras.layers.Embedding(input_dim=8192, output_dim=64)
        self.bidir_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64)
        )
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)  # no activation, from_logits=True used in loss

    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        :param inputs: A batch of sequences with padded tokens (batch_size, seq_len).
        :param training: Boolean, whether in training mode.
        :return: logits tensor of shape (batch_size, 1).
        """
        x = self.embedding(inputs)  # shape: (batch_size, seq_len, 64)
        x = self.bidir_lstm(x, training=training)  # shape: (batch_size, 128) because bidir doubles units
        x = self.dense1(x)  # shape: (batch_size, 64)
        logits = self.dense2(x)  # shape: (batch_size, 1)
        return logits

def my_model_function():
    """
    Creates an instance of MyModel with the assumed vocabulary size.
    Adjust the vocab_size in MyModel if necessary.
    """
    model = MyModel()
    return model

def GetInput():
    """
    Returns a batch of random integer sequences matching expected input.
    The input matches the shape and dtype expected by MyModel:
      - batch size 64 (from original code BATCH_SIZE)
      - variable sequence length padded batches; we fix length here for tensor shape
      - token IDs in range [0, vocab_size)
    Uses sequence length 100 as a reasonable guess to allow stable input.
    """
    batch_size = 64
    seq_length = 100  # fixed length for uniform input tensor; original batches padded dynamically
    vocab_size = 8192  # from subwords8k vocabulary size

    # Random integer tensor of (batch_size, seq_length)
    # Using tf.random.uniform with integer dtype is valid in TF >= 2.0
    inputs = tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return inputs


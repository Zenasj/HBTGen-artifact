# tf.random.uniform((64, 200), dtype=tf.int32) ← batch size 64, sequence length 200 as per example_input_batch shape

import tensorflow as tf

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size=106, embedding_dim=256, enc_units=1024, batch_sz=64):
    super(MyModel, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    # Embedding layer for input tokens, vocab size 106 inferred from error message
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    # LSTM layer with return sequences and states for Encoder
    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    """
    Forward pass:
    Inputs:
      x: tf.Tensor, shape (batch_sz, seq_len), integer token indices
      hidden: list of two tensors [h, c], each shape (batch_sz, enc_units)
    Outputs:
      output: LSTM output sequence, shape (batch_sz, seq_len, enc_units)
      h: hidden state (batch_sz, enc_units)
      c: cell state (batch_sz, enc_units)
    """
    x = self.embedding(x)
    output, h, c = self.lstm_layer(x, initial_state=hidden)
    return output, h, c

  def initialize_hidden_state(self):
    """
    Returns initial hidden states (h, c) for LSTM zeros.
    """
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]


def my_model_function():
  # Provide a default instantiation with common hyperparameters.
  # These can be modified as needed.
  return MyModel()

def GetInput():
  # Return a random integer tensor with shape (batch_sz=64, seq_len=200)
  # Values in [0, vocab_size-1) = [0, 106)
  batch_sz = 64
  seq_len = 200
  vocab_size = 106
  return tf.random.uniform(shape=(batch_sz, seq_len),
                           minval=0,
                           maxval=vocab_size,
                           dtype=tf.int32)


# tf.random.uniform((BATCH_SIZE, 1), dtype=tf.float32) ← comment line with inferred input shape for Decoder input

import tensorflow as tf
import numpy as np

# This code fuses two example models/discussions from the issue:
# 1) A seq2seq model with Bahdanau attention for machine translation 
# 2) A DDPG (actor-critic) reinforcement learning model using tf.distribute.MirroredStrategy but causing error due to distributed variable gradient handling.

# To illustrate a single tf.keras.Model wrapping the seq2seq Encoder-Decoder model,
# because the RL code is based on TF1 style tf.Session and low-level graph code.
# The error about DistributedVariable.handle inaccessible outside replica context comes 
# from that tf.Session + MirroredStrategy approach in TF 2.x eager environment.
#
# Here we implement the seq2seq model as MyModel, with expected input shape:
#     Input to encoder: (BATCH_SIZE, sequence_length)
#     Decoder call input: (BATCH_SIZE, 1)
#
# GetInput() returns a random batch of sequence input fitting the encoder input.
#
# This is a functional, self-contained example that can be compiled with XLA.

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: (batch_size, hidden size)
        # values: (batch_size, max_length, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)  # (batch_size, 1, hidden size)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))  # (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, max_length, 1)
        context_vector = attention_weights * values  # (batch_size, max_length, hidden size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, hidden size)
        return context_vector, attention_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.enc_units))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # x shape: (batch_size, 1) a token index for current decoder input timestep
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # (batch_size, 1, embedding_dim + hidden_size)
        output, state = self.gru(x)  # output: (batch_size, 1, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))  # (batch_size, dec_units)
        x = self.fc(output)  # (batch_size, vocab_size)
        return x, state, attention_weights


class MyModel(tf.keras.Model):
    """
    A fused model wrapping the Encoder-Decoder with attention.
    The forward call runs:
      - Encoder to get enc_output and enc_hidden
      - Decodes a single step given a decoder input and hidden state (for simplicity)
    """

    def __init__(self,
                 vocab_inp_size,
                 vocab_tar_size,
                 embedding_dim=256,
                 units=1024,
                 batch_size=192):
        super().__init__()
        self.batch_size = batch_size
        self.units = units
        self.encoder = Encoder(vocab_inp_size, embedding_dim, units)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, units)
        self.targ_lang_start_token = None  # to be set externally

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        Args:
          inputs: a tuple (inp, targ_lang_start_token), where
            inp: int32 tensor shape (batch_size, seq_len) encoder input tokens
            targ_lang_start_token: int scalar the index for <start> token in target language

        Returns:
          decoder output logits for one step: tensor shape (batch_size, vocab_tar_size)
        """
        inp, start_token_idx = inputs
        batch_size = tf.shape(inp)[0]

        enc_hidden = self.encoder.initialize_hidden_state(batch_size)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        # Prepare decoder input: a batch of <start> tokens
        dec_input = tf.expand_dims(tf.fill([batch_size], start_token_idx), 1)  # (batch_size, 1)

        # Decode one step (for example; full decoding loop can be external)
        predictions, _, _ = self.decoder(dec_input, dec_hidden, enc_output)

        return predictions


def my_model_function():
    # Provide default vocab sizes and batch size for demonstration.
    vocab_inp_size = 8000  # Assume input vocab size
    vocab_tar_size = 8000  # Assume target vocab size
    batch_size = 192
    model = MyModel(vocab_inp_size=vocab_inp_size,
                    vocab_tar_size=vocab_tar_size,
                    embedding_dim=256,
                    units=1024,
                    batch_size=batch_size)
    # Set dummy <start> token index (e.g., 1) for target language
    model.targ_lang_start_token = 1
    return model


def GetInput():
    """
    Returns a tuple (inp_tensor, start_token_index) that can be passed as inputs to MyModel.
    - inp_tensor: a tf.int32 tensor of shape (BATCH_SIZE, SEQ_LEN)
    - start_token_index: an integer scalar
    """
    BATCH_SIZE = 192
    SEQ_LEN = 15  # arbitrary sequence length
    VOCAB_SIZE = 8000

    # Random integer sequences as inputs, within vocab range [1, VOCAB_SIZE)
    inp_tensor = tf.random.uniform(
        shape=(BATCH_SIZE, SEQ_LEN),
        minval=1,
        maxval=VOCAB_SIZE,
        dtype=tf.int32
    )

    start_token_index = tf.constant(1, dtype=tf.int32)  # assume 1 is <start> token

    return (inp_tensor, start_token_index)


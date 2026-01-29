# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape is batch_size x sequence_length (variable length sequences)

import tensorflow as tf
import numpy as np

# We recreate the LSTM-based language model described in TF 1.x style, 
# modernized into a tf.keras.Model compatible with TF 2.20.0 XLA compilation.
# The model:
# - Embeds input word indices into vectors
# - Runs a multi-layer LSTM with dropout
# - Outputs logits for each timestep for predicting next token
# 
# Assumptions from original code:
# - batch size fixed at runtime (to simplify shape defs)
# - variable sequence length is supported (None), but for XLA compilation 
#   all inputs should have known sizes, so dummy input during GetInput uses fixed length
# - vocabulary size (= word count) is a parameter
# - embedding size = hidden_units = rnn_size (=128 by default)
# - 2 layers of BasicLSTMCell via tf.keras.layers.LSTM
# - Dropout 0.5 applied on LSTM layers output (training mode)
# - We fuse TF 1.x dynamic_rnn into Keras layers
# - Output logits shape (batch_size * seq_len, vocab_size) flattening time dimension

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, rnn_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=rnn_size,
                                                   mask_zero=True,
                                                   name="embedding")

        # Create stacked LSTM layers with dropout wrapped in Keras style
        self.lstm_layers = []
        for i in range(num_layers):
            # return_sequences=True to keep time dimension for next layer or output
            # dropout applied only on output connections (matching DropoutWrapper output_keep_prob)
            self.lstm_layers.append(
                tf.keras.layers.LSTM(rnn_size,
                                     return_sequences=True,
                                     recurrent_initializer='glorot_uniform',
                                     name=f'lstm_layer_{i}',
                                     dropout=dropout)
            )
        # Final dense layer for logits
        self.dense = tf.keras.layers.Dense(vocab_size, name="softmax_dense")

    def call(self, inputs, training=False):
        # inputs: shape (batch_size, seq_length) int32 word indices
        x = self.embedding(inputs)  # (batch_size, seq_length, rnn_size)
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)  # (batch_size, seq_length, rnn_size)
        # flatten seq dim for logits calculation
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, [batch_size * seq_len, self.rnn_size])
        logits = self.dense(x_reshaped)  # (batch_size * seq_len, vocab_size)
        probs = tf.nn.softmax(logits)
        return logits, probs


def my_model_function():
    # Here we must provide vocab_size for instantiation.
    # The original code used `len(words)` that comes from vocabulary
    # In absence of exact vocab size, assume 100 as a reasonable placeholder
    # User should replace this with actual vocabulary size as required.
    vocab_size = 100
    return MyModel(vocab_size=vocab_size, rnn_size=128, num_layers=2, dropout=0.5)


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Inputs: (batch_size, seq_length) integer word indices in range [0, vocab_size)
    batch_size = 8  # choose smaller batch size to avoid zero division issues as per comments
    seq_length = 50  # fixed sequence length for simplicity

    vocab_size = 100  # same as used in model
    # Generate random integer indices uniformly in [0, vocab_size)
    inputs = tf.random.uniform(shape=(batch_size, seq_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    return inputs


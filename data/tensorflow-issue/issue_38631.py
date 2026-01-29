# tf.random.uniform((B, None), dtype=tf.int64) ← Input is a batch of sequences of variable length (token ids)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=1000, hidden_size=32,
                 max_input_length=30, max_target_length=30):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, hidden_size, input_length=max_input_length)
        self.encoder_rnn = layers.RNN(layers.GRUCell(hidden_size))
        self.repeat_vector = layers.RepeatVector(max_target_length)
        self.decoder_rnn = layers.RNN(layers.GRUCell(hidden_size), return_sequences=True)
        self.time_dist_relu = layers.TimeDistributed(layers.Dense(hidden_size, activation='relu'))
        self.time_dist_out = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))

    def call(self, inputs, training=False):
        """
        inputs: tensor of shape [batch_size, seq_len] with dtype int64 (token ids)
        Returns:
            tensor shape [batch_size, max_target_length, vocab_size]
            with softmax probabilities per timestep
        """
        x = self.embedding(inputs)              # [B, T_in, hidden_size]
        x = self.encoder_rnn(x)                 # [B, hidden_size] - summary vector from encoder
        x = self.repeat_vector(x)               # [B, max_target_length, hidden_size]
        x = self.decoder_rnn(x)                 # [B, max_target_length, hidden_size]
        x = self.time_dist_relu(x)              # [B, max_target_length, hidden_size]
        outputs = self.time_dist_out(x)         # [B, max_target_length, vocab_size], softmax
        return outputs


def my_model_function():
    # These params are from the original word-completion example defaults:
    vocab_size = 1000
    hidden_size = 32
    max_input_length = 30
    max_target_length = 30
    return MyModel(vocab_size=vocab_size,
                   hidden_size=hidden_size,
                   max_input_length=max_input_length,
                   max_target_length=max_target_length)


def GetInput():
    # Generate a batch of inputs matching the model's expected input shape and type.
    # Since input shape is (batch_size, sequence_length), both variable lengths but max 30 here.
    batch_size = 20
    seq_len = 30
    # Generate random int64 tensors for token IDs in [0, vocab_size-1]
    vocab_size = 1000
    # Use tf.random.uniform to generate a batch of sequences of ints in [0, vocab_size)
    return tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int64)


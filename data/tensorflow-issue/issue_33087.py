# tf.random.uniform((batch_size, seq_len, vocab_size), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=162, units=512, dropout=0.0):
        super().__init__()
        # Encoder: Bidirectional GRU with return_sequences and return_state
        self.encoder_bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units,
                                return_sequences=True,
                                return_state=True,
                                dropout=dropout))
        # Decoder: GRU with double units (because encoder states concatenated)
        self.decoder_gru = tf.keras.layers.GRU(units * 2,
                                               return_sequences=True,
                                               return_state=True,
                                               dropout=dropout)
        # Attention layer
        self.attention = tf.keras.layers.Attention()
        # Dense projection wrapped in TimeDistributed for output
        self.dense_time = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(vocab_size, activation="softmax"))
        self.vocab_size = vocab_size
        self.units = units

    def call(self, inputs, training=False):
        """
        inputs: tuple/list of two tensors (encoder_inputs, decoder_inputs)
          encoder_inputs: shape (batch_size, enc_seq_len, vocab_size)
          decoder_inputs: shape (batch_size, dec_seq_len, vocab_size)
        returns:
          decoder_pred: shape (batch_size, dec_seq_len, vocab_size), softmax output
        """
        encoder_inputs, decoder_inputs = inputs

        # Encoder forward pass
        encoder_out, fwd_state, back_state = self.encoder_bigru(encoder_inputs,
                                                                training=training)
        # Concatenate forward and backward states for decoder initial state
        encoder_states = tf.keras.layers.Concatenate(axis=-1)([fwd_state, back_state])

        # Decoder forward pass, using encoder_states as initial_state
        decoder_out, _ = self.decoder_gru(decoder_inputs,
                                          initial_state=encoder_states,
                                          training=training)

        # Compute attention between decoder output and encoder output
        attn_out = self.attention([decoder_out, encoder_out])

        # Concatenate decoder GRU output with attention output
        decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_out, attn_out])

        # Final output projection with softmax over vocab
        decoder_pred = self.dense_time(decoder_concat_input)

        return decoder_pred


def my_model_function():
    # Return an instance of MyModel with default parameters simulating the issue's setup
    return MyModel(vocab_size=162, units=512, dropout=0.0)


def GetInput():
    # Generate a tuple of two inputs matching the expected input shapes:
    # - encoder_inputs shape: (batch_size, enc_seq_len, vocab_size)
    # - decoder_inputs shape: (batch_size, dec_seq_len, vocab_size)
    # Use batch_size=128, enc_seq_len=128, dec_seq_len=128 as example from issue
    batch_size = 128
    enc_seq_len = 128
    dec_seq_len = 128
    vocab_size = 162

    encoder_inputs = tf.random.uniform(
        (batch_size, enc_seq_len, vocab_size), dtype=tf.float32)
    decoder_inputs = tf.random.uniform(
        (batch_size, dec_seq_len, vocab_size), dtype=tf.float32)

    return encoder_inputs, decoder_inputs


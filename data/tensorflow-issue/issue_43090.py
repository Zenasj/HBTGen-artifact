# tf.random.uniform((123, 45, 4), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n_input=4, n_output=1, n_units=64):
        super().__init__()
        # Encoder LSTM returns states for the input sequence (None timestep dimension)
        self.encoder_lstm = tf.keras.layers.LSTM(n_units, return_state=True)
        # Decoder LSTM returns sequence output and states; decoder input shape: (None, n_output)
        self.decoder_lstm = tf.keras.layers.LSTM(n_units, return_sequences=True, return_state=True)
        # Output dense layer with softmax activation (as in original code)
        self.dense = tf.keras.layers.Dense(n_output, activation='softmax')

    def call(self, inputs, training=False):
        """
        inputs: tuple of (encoder_inputs, decoder_inputs)
          encoder_inputs shape: (batch, seq_len_encoder, n_input) - here seq_len_encoder=45, n_input=4
          decoder_inputs shape: (batch, seq_len_decoder, n_output) - seq_len_decoder=45, n_output=1

        Returns output of shape (batch, seq_len_decoder, n_output)
        """
        encoder_inputs, decoder_inputs = inputs
        # Encoder LSTM processes the full encoder input sequence and returns states
        _, state_h, state_c = self.encoder_lstm(encoder_inputs, training=training)
        encoder_states = [state_h, state_c]

        # Decoder LSTM processes decoder_inputs, initialized with encoder states
        decoder_outputs, _, _ = self.decoder_lstm(decoder_inputs, initial_state=encoder_states, training=training)

        # Dense layer with softmax activation on decoder outputs
        output = self.dense(decoder_outputs)
        return output


def my_model_function():
    # Instantiate MyModel with parameter values inferred from original code:
    # n_input=4 (input features), n_output=1 (output features), n_units=64 (chosen default)
    return MyModel(n_input=4, n_output=1, n_units=64)


def GetInput():
    # Return example inputs compatible with MyModel:
    # encoder_inputs shape: (batch_size=123, seq_len=45, n_input=4)
    # decoder_inputs shape: (batch_size=123, seq_len=45, n_output=1)
    encoder_inputs = tf.random.uniform(shape=(123, 45, 4), dtype=tf.float32)
    decoder_inputs = tf.random.uniform(shape=(123, 45, 1), dtype=tf.float32)
    return (encoder_inputs, decoder_inputs)


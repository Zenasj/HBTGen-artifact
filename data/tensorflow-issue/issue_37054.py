# tf.random.uniform((B,)) ← Input shape is (batch_size, max_input_length) and (batch_size, max_output_length) integers (token IDs)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, GRU, concatenate, dot, Activation,
    TimeDistributed, Dense
)
from tensorflow.keras import Model
import os

# Placeholder for build_params function since it is not defined in the original code.
# Assumes build_params returns a dictionary with relevant keys:
# 'input_encoding', 'input_decoding', 'input_dict_size', 'output_encoding',
# 'output_decoding', 'output_dict_size', 'max_input_length', 'max_output_length'.
def build_params(params_path='params'):
    # In a real scenario, this function would load params from a file or construct them.
    # Here we define example defaults for dimensions and encodings.
    params = {
        'input_encoding': {'token_to_id': {}},  # dummy dict
        'input_decoding': {'id_to_token': {}},
        'input_dict_size': 50,    # size of source vocab
        'output_encoding': {'token_to_id': {}},
        'output_decoding': {'id_to_token': {}},
        'output_dict_size': 39,   # size of target vocab
        'max_input_length': 10,   # maximum length of input sequences
        'max_output_length': 10,  # maximum length of output sequences
    }
    return params

class MyModel(tf.keras.Model):
    def __init__(self, params_path='params', enc_lstm_units=128, unroll=True, use_gru=False, optimizer='adam', display_summary=True):
        super().__init__()
        # Load parameters
        self.params = build_params(params_path)
        self.enc_lstm_units = enc_lstm_units
        self.unroll = unroll
        self.use_gru = use_gru
        self.optimizer = optimizer
        self.display_summary = display_summary

        # Extract parameters for ease of access
        self.input_dict_size = self.params['input_dict_size']
        self.output_dict_size = self.params['output_dict_size']
        self.max_input_length = self.params['max_input_length']
        self.max_output_length = self.params['max_output_length']

        # Define layers that do not depend on input shape
        self.encoder_embedding = Embedding(
            self.input_dict_size,
            self.enc_lstm_units,
            input_length=self.max_input_length,
            mask_zero=True
        )
        self.decoder_embedding = Embedding(
            self.output_dict_size,
            2 * self.enc_lstm_units,
            input_length=self.max_output_length,
            mask_zero=True
        )

        if not self.use_gru:
            self.encoder_rnn = Bidirectional(
                LSTM(self.enc_lstm_units, return_sequences=True, return_state=True, unroll=self.unroll),
                merge_mode='concat'
            )
            self.decoder_rnn = LSTM(
                2 * self.enc_lstm_units, return_sequences=True, unroll=self.unroll
            )
        else:
            self.encoder_rnn = Bidirectional(
                GRU(self.enc_lstm_units, return_sequences=True, return_state=True, unroll=self.unroll),
                merge_mode='concat'
            )
            self.decoder_rnn = GRU(
                2 * self.enc_lstm_units, return_sequences=True, unroll=self.unroll
            )

        # Attention and output layers
        self.attention_activation = Activation('softmax', name='attention')
        self.output_dense_1 = TimeDistributed(Dense(self.enc_lstm_units, activation='tanh'))
        self.output_dense_2 = TimeDistributed(Dense(self.output_dict_size, activation='softmax'))

        # Build the model within the class but separate from tf.function call
        # We will recreate the call logic inside call()

    def call(self, inputs, training=False):
        """
        Forward pass
        inputs: tuple of (encoder_input, decoder_input)
        encoder_input: int tensor of shape (batch_size, max_input_length)
        decoder_input: int tensor of shape (batch_size, max_output_length)
        """
        encoder_input, decoder_input = inputs

        # Encoder embedding + encoding
        enc_emb = self.encoder_embedding(encoder_input)  # (B, max_input_length, enc_lstm_units)
        # Encoder RNN returns outputs and states depending on use_gru or not
        if not self.use_gru:
            # LSTM returns (outputs, forward_h, forward_c, backward_h, backward_c)
            encoder_outs, forward_h, forward_c, backward_h, backward_c = self.encoder_rnn(enc_emb, training=training)
            encoder_h = concatenate([forward_h, backward_h])
            encoder_c = concatenate([forward_c, backward_c])
            # Decoder RNN run with initial states
            decoder_emb = self.decoder_embedding(decoder_input)  # (B, max_output_length, 2*enc_lstm_units)
            decoder_outs = self.decoder_rnn(decoder_emb, initial_state=[encoder_h, encoder_c], training=training)
        else:
            # GRU returns (outputs, forward_h, backward_h)
            encoder_outs, forward_h, backward_h = self.encoder_rnn(enc_emb, training=training)
            encoder_h = concatenate([forward_h, backward_h])
            decoder_emb = self.decoder_embedding(decoder_input)
            decoder_outs = self.decoder_rnn(decoder_emb, initial_state=encoder_h, training=training)

        # Attention mechanism (Luong style)
        attention = dot([decoder_outs, encoder_outs], axes=[2, 2])
        attention = self.attention_activation(attention)
        context = dot([attention, encoder_outs], axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder_outs])

        # Output projection
        output = self.output_dense_1(decoder_combined_context)
        output = self.output_dense_2(output)

        return output

def my_model_function(params_path='params', enc_lstm_units=128, unroll=True, use_gru=False, optimizer='adam', display_summary=False):
    # Instantiate the model inside the TPU strategy scope for TPU compatibility
    
    # Setup TPU resolver and strategy (only if TPU environment detected)
    resolver = None
    strategy = None
    tpu_address = os.environ.get('COLAB_TPU_ADDR', None)
    if tpu_address:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        # fallback strategy for non-TPU environments
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = MyModel(
            params_path=params_path,
            enc_lstm_units=enc_lstm_units,
            unroll=unroll,
            use_gru=use_gru,
            optimizer=optimizer,
            display_summary=display_summary
        )
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if display_summary:
        # Build concrete input shapes for summary display
        encoder_shape = (None, model.max_input_length)
        decoder_shape = (None, model.max_output_length)
        model.build(input_shape=[encoder_shape, decoder_shape])
        model.summary()

    return model

def GetInput():
    """
    Returns:
        Tuple of (encoder_input, decoder_input), both integer tensors,
        compatible as inputs to MyModel call method.
    Shapes:
        encoder_input: (batch_size, max_input_length)
        decoder_input: (batch_size, max_output_length)
    Values are integers in the expected input dictionary ranges.
    """
    params = build_params()
    batch_size = 4  # Example batch size for randomness
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']
    input_dict_size = params['input_dict_size']
    output_dict_size = params['output_dict_size']

    # Random integers in the range [1, dict_size-1] because 0 is reserved for mask_zero
    encoder_input = tf.random.uniform(
        shape=(batch_size, max_input_length),
        minval=1, maxval=input_dict_size,
        dtype=tf.int32
    )
    decoder_input = tf.random.uniform(
        shape=(batch_size, max_output_length),
        minval=1, maxval=output_dict_size,
        dtype=tf.int32
    )

    return (encoder_input, decoder_input)


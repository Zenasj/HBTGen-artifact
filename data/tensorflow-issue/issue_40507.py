# tf.random.uniform((batch_size, input_len), dtype=tf.int32) and tf.random.uniform((batch_size, output_len), dtype=tf.int32)
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

# For clarity, we define some constants inferred from the issue's code
START_ID = 1  # Assumed start token id
EOS_ID = 2    # Assumed end token id

class MyModel(tf.keras.Model):
    def __init__(self,
                 vocab_size=10000,
                 input_len=20,
                 output_len=20,
                 batch_size=1,
                 rnn_units=64,
                 dense_units=64,
                 embedding_dim=256,
                 beam_width=3):
        super(MyModel, self).__init__()
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.embedding_dim = embedding_dim
        self.beam_width = beam_width

        # Encoder embedding and LSTM
        self.encoder_embedding = layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.input_len)
        self.encoder_rnn = layers.LSTM(self.rnn_units, return_sequences=True, return_state=True)

        # Decoder embedding and LSTMCell
        self.decoder_embedding = layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.output_len)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(self.rnn_units)

        # Attention mechanism (Luong attention)
        self.attention_mechanism = tfa.seq2seq.LuongAttention(self.dense_units)

        # Wrap LSTMCell with AttentionWrapper
        self.rnn_cell = self.build_rnn_cell(self.batch_size)

        # Output projection
        self.dense_layer = layers.Dense(self.vocab_size)

        # BeamSearchDecoder initialization with critical params
        # embedding_fn uses tf.gather on the decoder embedding weights to sidestep unsupported tf.nn.embedding_lookup ops in TFLite
        self.inference_decoder = tfa.seq2seq.BeamSearchDecoder(
            cell=self.rnn_cell,
            beam_width=self.beam_width,
            output_layer=self.dense_layer,
            embedding_fn=lambda ids: tf.gather(self.decoder_embedding.variables[0], ids),
            coverage_penalty_weight=0.0,
            dynamic=False,
            parallel_iterations=1,
            maximum_iterations=self.output_len
        )

    def build_rnn_cell(self, batch_size):
        # Wrap the LSTMCell with AttentionWrapper using the Luong attention mechanism
        # The attention is wrapped with the correct memory shape to match encoder outputs' time dimension * beam width
        # Note: batch_size might be tiled in beam search, so attention mechanism setup will handle it
        cell = self.decoder_rnncell
        attention_cell = tfa.seq2seq.AttentionWrapper(
            cell,
            self.attention_mechanism,
            attention_layer_size=self.dense_units,
            alignment_history=False,
            name="attention_wrapper"
        )
        return attention_cell

    def build_decoder_initial_state(self, size, encoder_state, Dtype):
        # encoder_state is a list [state_h, state_c] from encoder LSTM
        # Beam search decoder initial state expects an AttentionWrapperState with:
        # cell_state: LSTMStateTuple with cell and hidden states (tiled),
        # attention: tensor (initially zeros),
        # time: scalar time step,
        # alignments: zero tensor,
        # alignment_history: None (disabled),
        # and possibly other fields.

        # Tile the encoder state for beam_width if needed (size could be batch_size * beam_width)
        # 'size' argument represents batch_size * beam_width

        # The AttentionWrapperState requires creating LSTMStateTuples for cell_state
        # We tile the encoder cell states accordingly.
        cell_state = tf.keras.layers.Lambda(
            lambda x: tfa.seq2seq.tile_batch(x, multiplier=self.beam_width)
        )(encoder_state)

        # cell_state is a tuple (h, c) in encoder_state, but our encoder uses LSTM -- unpack it:
        # In call(), the encoder outputs state_h and state_c; here we expect encoder_state has [state_h, state_c]
        # Make sure these are tensors, then tile them.
        # We manually tile them here since Lambda may behave unexpectedly:
        tiled_h = tfa.seq2seq.tile_batch(encoder_state[0], multiplier=self.beam_width)
        tiled_c = tfa.seq2seq.tile_batch(encoder_state[1], multiplier=self.beam_width)
        cell_state = tf.keras.layers.LSTMStateTuple(tiled_h, tiled_c)

        attn_mech = self.attention_mechanism
        # Setup attention mechanism state: initial alignments and attention = zeros
        # For BeamSearchDecoder, these are handled internally,
        # but AttentionWrapperState expects zeroed tensors matching batch_size*beam_width

        # Create initial AttentionWrapperState
        initial_state = tfa.seq2seq.AttentionWrapperState(
            cell_state=cell_state,
            time=tf.constant(0, dtype=tf.int32),
            attention=tf.zeros([size, self.dense_units], dtype=Dtype),
            alignments=tf.zeros([size, attn_mech.memory_time], dtype=Dtype) if hasattr(attn_mech, 'memory_time') else None,
            alignment_history=None,
        )
        return initial_state

    def call(self, inputs, training=None, mask=None):
        """
        inputs: tuple of two tensors:
          - inputs[0]: encoder input ids (batch_size, input_len) integers
          - inputs[1]: decoder input ids (batch_size, output_len) integers
        """
        encoder_inputs, decoder_inputs = inputs

        # 1) Encode inputs
        encoder_embedded = self.encoder_embedding(encoder_inputs)  # (batch_size, input_len, embedding_dim)
        encoder_outputs, state_h, state_c = self.encoder_rnn(encoder_embedded, training=training)
        # encoder_outputs: (batch_size, input_len, rnn_units)

        # 2) Setup attention mechanism on encoder outputs
        self.attention_mechanism.setup_memory(encoder_outputs)

        # 3) Tile encoder outputs and states because of beam search
        encoder_outputs_tiled = tfa.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_width)
        state_h_tiled = tfa.seq2seq.tile_batch(state_h, multiplier=self.beam_width)
        state_c_tiled = tfa.seq2seq.tile_batch(state_c, multiplier=self.beam_width)

        # 4) Prepare start tokens - int tensor shaped [batch_size]
        # Since original code uses tf.fill([1], START_ID), but with batch_size:
        batch_size = tf.shape(encoder_inputs)[0]
        start_tokens = tf.fill([batch_size], START_ID)

        # 5) Build the initial state for the decoder
        initial_state = self.build_decoder_initial_state(
            size=batch_size * self.beam_width,
            encoder_state=[state_h_tiled, state_c_tiled],
            Dtype=encoder_outputs.dtype
        )

        # 6) Run inference decoder
        # The BeamSearchDecoder expects its call to pass embedding=None and use the embedding_fn configured
        # Also end_token indicates when to stop decoding
        final_outputs, final_states, final_sequence_lengths = self.inference_decoder(
            embedding=None,
            start_tokens=start_tokens,
            end_token=EOS_ID,
            initial_state=initial_state
        )

        # final_outputs.predicted_ids shape: (batch_size, max_time, beam_width)
        # For simplicity, return predicted_ids directly
        return final_outputs.predicted_ids


def my_model_function():
    # Returns an instance of MyModel with typical/default parameters
    # These defaults match the example values from the issue
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the inputs expected by MyModel

    # From the model, inputs is a tuple of two integer tensors:
    # (encoder_input_ids, decoder_input_ids)
    # Shapes: (batch_size, input_len), (batch_size, output_len)
    # For demonstration we pick batch_size=1, input_len=20, output_len=20 as defaults.

    batch_size = 1
    input_len = 20
    output_len = 20
    vocab_size = 10000  # To match default model vocab size

    encoder_input = tf.random.uniform(
        shape=(batch_size, input_len), minval=3, maxval=vocab_size, dtype=tf.int32
    )
    decoder_input = tf.random.uniform(
        shape=(batch_size, output_len), minval=3, maxval=vocab_size, dtype=tf.int32
    )
    return (encoder_input, decoder_input)


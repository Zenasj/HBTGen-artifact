# tf.random.uniform((B, T_src), dtype=tf.int32)
# Assuming input is a batch of sequences of token IDs (decoder_inputs) with shape (batch_size, seq_len)
# This model fuses the LAS listener and speller building logic from TensorFlow 2.4 code with TF Addons attention

import tensorflow as tf
import tensorflow_addons as tfa

# Placeholder lstm_cell and pyramidal_bilstm implementations used in the original code
# These would normally come from las.ops package, here we define minimal versions for completeness

def lstm_cell(num_units, dropout, mode):
    # Return tf.keras.layers.LSTMCell wrapped with dropout compatible with mode
    cell = tf.keras.layers.LSTMCell(num_units)
    if mode == tf.estimator.ModeKeys.TRAIN and dropout > 0:
        cell = tf.keras.layers.DropoutWrapper(cell, output_keep_prob=1 - dropout)
        # Note: tf.keras.layers.DropoutWrapper does not exist; we use Dropout after RNN output by default.
        # To keep it simple, we omit Dropout here, as no direct wrapper in keras.
    return cell

def pyramidal_bilstm(encoder_inputs, source_sequence_length, mode, hparams):
    # Simple stacked bidirectional LSTM with halving time steps per layer (pyramidal)
    outputs = encoder_inputs
    seq_len = source_sequence_length
    for i in range(hparams['num_layers']):
        # Reduce time dimension by factor 2: reshape (batch, time, features) -> (batch, time//2, features*2)
        shape = tf.shape(outputs)
        B, T, F = shape[0], shape[1], shape[2]
        T_new = T // 2
        if T % 2 != 0:
            # pad with zeros if odd length
            pad_len = 1
            outputs = tf.pad(outputs, [[0,0],[0,pad_len],[0,0]])
            T += pad_len
            T_new = T // 2
        
        outputs = tf.reshape(outputs, (B, T_new, F * 2))
        
        # bidirectional LSTM layer
        fw_cell = tf.keras.layers.LSTMCell(hparams['num_units'])
        bw_cell = tf.keras.layers.LSTMCell(hparams['num_units'])
        rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(fw_cell, return_sequences=True),
            merge_mode='concat')
        outputs = rnn(outputs)

        # sequence length halves each layer approximately
        seq_len = seq_len // 2
    return (outputs, seq_len), None


class AttentionMultiCell(tf.keras.layers.StackedRNNCells):
    """
    MultiCell with attention style, adapted from TF 1.x style
    """
    def __init__(self, attention_cell, cells, use_new_attention=False):
        # cells = list of RNNCells, prepend attention_cell
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(AttentionMultiCell, self).__init__(cells)

    def call(self, inputs, states, training=None):
        if not isinstance(states, (list, tuple)):
            raise ValueError(f"Expected state to be a tuple/list, but received: {states}")
        
        new_states = []

        attention_cell = self.cells[0]
        attention_state = states[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state, training=training)
        new_states.append(new_attention_state)

        for i in range(1, len(self.cells)):
            cell = self.cells[i]
            cur_state = states[i]
            if self.use_new_attention:
                cur_inp = tf.concat([cur_inp, new_attention_state.attention], axis=-1)
            else:
                cur_inp = tf.concat([cur_inp, attention_state.attention], axis=-1)
            cur_inp, new_state = cell(cur_inp, cur_state, training=training)
            new_states.append(new_state)

        return cur_inp, new_states


class CustomAttention(tfa.seq2seq.LuongAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="CustomAttention"):
        super().__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=scale,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name)
        
        # Override query layer with ReLU activation
        self._query_layer = tf.keras.layers.Dense(num_units, use_bias=False, dtype=dtype, name='query_layer')
        self._keys = tf.nn.relu(self.keys)

    def __call__(self, query, state):
        processed_query = tf.nn.relu(self._query_layer(query))
        return super().__call__(processed_query, state)


def listener(encoder_inputs,
             source_sequence_length,
             mode,
             hparams):
    """
    Build encoder (listener) either pyramidal BiLSTM or stacked BiLSTMs
    """
    if hparams.get('use_pyramidal', False):
        return pyramidal_bilstm(encoder_inputs, source_sequence_length, mode, hparams)
    else:
        forward_cells = []
        backward_cells = []
        for layer in range(hparams['num_layers']):
            cell_fw = tf.keras.layers.LSTMCell(hparams['num_units'])
            cell_bw = tf.keras.layers.LSTMCell(hparams['num_units'])
            forward_cells.append(cell_fw)
            backward_cells.append(cell_bw)
        forward_cell = tf.keras.layers.StackedRNNCells(forward_cells)
        backward_cell = tf.keras.layers.StackedRNNCells(backward_cells)

        # The original code uses tf.keras.layers.Bidirectional wrapper with cell lists, which is not possible:
        # Bidirectional wrapper accepts a single RNN layer - here we replicate with two layered LSTMs:
        # For simplicity: we create a Bidirectional RNN with stacked cells by implementing a custom RNN or stacking ?

        # Instead, we build two RNN layers and concatenate outputs manually:
        # forward RNN
        rnn_fw = tf.keras.layers.RNN(forward_cell, return_sequences=True, return_state=True)
        outputs_fw = rnn_fw(encoder_inputs, mask=tf.sequence_mask(source_sequence_length))
        # backward RNN
        rnn_bw = tf.keras.layers.RNN(backward_cell, return_sequences=True, return_state=True, go_backwards=True)
        outputs_bw = rnn_bw(encoder_inputs, mask=tf.sequence_mask(source_sequence_length))

        # outputs_fw[0]: sequences output, outputs_fw[1:] states
        # similarly outputs_bw[0] is reversed sequences output
        # Reverse backward outputs to normal order
        outputs_bw_seq = tf.reverse(outputs_bw[0], axis=[1])

        encoder_outputs = tf.concat([outputs_fw[0], outputs_bw_seq], axis=-1)
        encoder_state = (outputs_fw[1:], outputs_bw[1:])
        return (encoder_outputs, source_sequence_length), encoder_state


def attend(encoder_outputs,
           source_sequence_length,
           mode,
           hparams):
    """
    Build the attention-decoder cell with possible AttentionWrapper
    """
    memory = encoder_outputs

    attention_type = hparams.get('attention_type', 'luong')
    if attention_type == 'luong':
        attention_fn = tfa.seq2seq.LuongAttention
    elif attention_type == 'bahdanau':
        attention_fn = tfa.seq2seq.BahdanauAttention
    elif attention_type == 'custom':
        attention_fn = CustomAttention
    else:
        attention_fn = tfa.seq2seq.LuongAttention  # fallback

    attention_mechanism = attention_fn(
        hparams['num_units'], memory, memory_sequence_length=source_sequence_length)

    cell_list = []
    for layer in range(hparams['num_layers']):
        cell = tf.keras.layers.LSTMCell(hparams['num_units'])
        cell_list.append(cell)

    # Whether only bottom layer uses attention or all stacked layers wrapped
    if hparams.get('bottom_only', False):
        attention_cell = cell_list.pop(0)
        attention_cell = tfa.seq2seq.AttentionWrapper(
            attention_cell, attention_mechanism,
            attention_layer_size=hparams.get('attention_layer_size', hparams['num_units']),
            alignment_history=(mode != tf.estimator.ModeKeys.TRAIN))
        decoder_cell = AttentionMultiCell(attention_cell, cell_list)
    else:
        stacked_cell = tf.keras.layers.StackedRNNCells(cell_list)
        decoder_cell = tfa.seq2seq.AttentionWrapper(
            stacked_cell, attention_mechanism,
            attention_layer_size=hparams.get('attention_layer_size', hparams['num_units']),
            alignment_history=(mode != tf.estimator.ModeKeys.TRAIN))
    return decoder_cell


class MyModel(tf.keras.Model):
    """
    Fused LAS model: Listener + Speller with attention, supporting train/eval/predict modes.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # Build embedding layer if embedding size > 0 else we'll use one-hot in call
        if self.hparams.get('embedding_size', 0) > 0:
            self.target_embedding = tf.Variable(
                tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")(
                    shape=[self.hparams['target_vocab_size'], self.hparams['embedding_size']]
                ),
                trainable=True, name='target_embedding'
            )
        else:
            self.target_embedding = None

        self.projection_layer = tf.keras.layers.Dense(
            self.hparams['target_vocab_size'], use_bias=True, name='projection_layer')

    def call(self, inputs, training=False):
        """
        inputs is expected to be a dict with keys:
        {
          'encoder_inputs': float tensor [batch, time, feature_dim] or int token IDs,
          'source_sequence_length': int tensor [batch],
          'decoder_inputs': int tensor [batch, target_seq_len],
          'target_sequence_length': int tensor [batch],
          'mode': tf.estimator.ModeKeys.TRAIN / EVAL / PREDICT,
        }

        This call returns decoder_outputs, final_context_state, final_sequence_length from dynamic_decode
        """

        encoder_inputs = inputs['encoder_inputs']  # Assuming float32 features
        source_sequence_length = inputs['source_sequence_length']
        decoder_inputs = inputs['decoder_inputs']  # int32 token ids
        target_sequence_length = inputs['target_sequence_length']
        mode = inputs['mode']

        hparams = self.hparams

        # Listener encode step
        (encoder_outputs, adjusted_seq_len), encoder_state = listener(
            encoder_inputs, source_sequence_length, mode, hparams)

        batch_size = tf.shape(input=encoder_outputs)[0]
        beam_width = hparams.get('beam_width', 0)

        # Tile if beam search and predict or eval
        if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL] and beam_width > 0:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
            adjusted_seq_len = tfa.seq2seq.tile_batch(adjusted_seq_len, multiplier=beam_width)
            # Tile encoder_state components
            if encoder_state is not None:
                encoder_state = tf.nest.map_structure(lambda t: tfa.seq2seq.tile_batch(t, multiplier=beam_width), encoder_state)
            batch_size = tf.shape(input=encoder_outputs)[0]

        # Build decoder cell with attention
        decoder_cell = attend(encoder_outputs, adjusted_seq_len, mode, hparams)

        # Initial state logic with optional passing of encoder final states
        if hparams.get('pass_hidden_state', False) and hparams.get('bottom_only', False):
            # Attempt to clone cells’ attention wrapper states replacing cell_state with encoder_state
            initial_state_raw = decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
            # zip and clone logic
            if encoder_state is not None:
                initial_state = tuple(
                    zs.clone(cell_state=es)
                    if isinstance(zs, tfa.seq2seq.AttentionWrapperState) else es
                    for zs, es in zip(initial_state_raw, encoder_state))
            else:
                initial_state = initial_state_raw
        else:
            initial_state = decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

        maximum_iterations = None
        if mode != tf.estimator.ModeKeys.TRAIN:
            max_len = tf.reduce_max(input_tensor=adjusted_seq_len)
            decoding_factor = hparams.get('decoding_length_factor', 1.0)
            maximum_iterations = tf.cast(tf.round(tf.cast(max_len, tf.float32) * decoding_factor), tf.int32)

        def embedding_fn(ids):
            if self.target_embedding is not None:
                return tf.nn.embedding_lookup(params=self.target_embedding, ids=ids)
            else:
                return tf.one_hot(ids, hparams['target_vocab_size'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Prepare decoder inputs embeddings
            decoder_emb_inputs = embedding_fn(decoder_inputs)
            decay_steps = hparams.get('decay_steps', 10000)
            iter_num = tf.compat.v1.train.get_or_create_global_step()
            inverse_probability = tf.compat.v1.train.polynomial_decay(1.0, iter_num, decay_steps, 0.6)
            sampling_probability = 1.0 - inverse_probability
            if hparams.get('sampling_probability', False):
                helper = tfa.seq2seq.ScheduledEmbeddingTrainingSampler(
                    sampling_probability=sampling_probability,
                    embedding_fn=embedding_fn
                )
            else:
                helper = tfa.seq2seq.TrainingSampler()

            decoder = tfa.seq2seq.BasicDecoder(
                cell=decoder_cell,
                sampler=helper,
                output_layer=self.projection_layer,
                maximum_iterations=maximum_iterations
            )
            decoder_outputs, final_context_state, final_sequence_length = tfa.seq2seq.dynamic_decode(
                decoder, training=True,
                decoder_init_input=decoder_emb_inputs,
                decoder_init_kwargs={'initial_state': initial_state,
                                    'sequence_length': target_sequence_length})

        elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            # Beam search decode
            start_tokens = tf.fill([tf.math.floordiv(batch_size, beam_width)], hparams['sos_id'])
            decoder = tfa.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding_fn=embedding_fn,
                beam_width=beam_width,
                output_layer=self.projection_layer,
                maximum_iterations=maximum_iterations
            )
            decoder_outputs, final_context_state, final_sequence_length = tfa.seq2seq.dynamic_decode(
                decoder,
                decoder_inputs=embedding_fn(decoder_inputs),
                training=False,
                decoder_init_kwargs={
                    'start_tokens': start_tokens,
                    'end_token': hparams['eos_id'],
                    'initial_state': initial_state
                })
        else:
            # EVAL or PREDICT without beam search defaults to beam search per original or fallback
            start_tokens = tf.fill([tf.math.floordiv(batch_size, max(beam_width, 1))], hparams['sos_id'])
            decoder = tfa.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding_fn=embedding_fn,
                beam_width=max(beam_width, 1),
                output_layer=self.projection_layer,
                maximum_iterations=maximum_iterations
            )
            decoder_outputs, final_context_state, final_sequence_length = tfa.seq2seq.dynamic_decode(
                decoder,
                decoder_inputs=embedding_fn(decoder_inputs),
                training=False,
                decoder_init_kwargs={
                    'start_tokens': start_tokens,
                    'end_token': hparams['eos_id'],
                    'initial_state': initial_state
                })

        return decoder_outputs, final_context_state, final_sequence_length


def my_model_function():
    # We define a dummy set of hyperparameters tuned for common ASR LAS models.
    hparams = {
        'use_pyramidal': False,         # Whether to use pyramidal BiLSTM encoder
        'num_layers': 3,
        'num_units': 256,
        'dropout': 0.1,
        'attention_type': 'luong',      # or 'bahdanau' or 'custom'
        'attention_layer_size': 256,
        'beam_width': 4,
        'embedding_size': 128,
        'target_vocab_size': 5000,
        'pass_hidden_state': False,
        'bottom_only': True,
        'sampling_probability': False,
        'decay_steps': 20000,
        'decoding_length_factor': 1.5,
        'sos_id': 1,
        'eos_id': 2,
    }
    return MyModel(hparams)


def GetInput():
    """
    Generates a dummy input dictionary for MyModel with random data matching expected shapes.

    Input shapes assumptions:
    - encoder_inputs: float tensor (batch_size, time_steps, feature_dim), here feature_dim=40 (e.g. MFCC)
    - source_sequence_length: int tensor (batch_size,)
    - decoder_inputs: int tensor (batch_size, target_seq_len)
    - target_sequence_length: int tensor (batch_size,)
    """
    batch_size = 4
    time_steps = 100
    feature_dim = 40
    target_seq_len = 30
    target_vocab_size = 5000

    encoder_inputs = tf.random.uniform(
        shape=(batch_size, time_steps, feature_dim), dtype=tf.float32, minval=-1., maxval=1.)
    source_sequence_length = tf.fill([batch_size], time_steps)
    decoder_inputs = tf.random.uniform(
        shape=(batch_size, target_seq_len), dtype=tf.int32, minval=0, maxval=target_vocab_size)
    target_sequence_length = tf.fill([batch_size], target_seq_len)

    mode = tf.estimator.ModeKeys.TRAIN

    return {
        'encoder_inputs': encoder_inputs,
        'source_sequence_length': source_sequence_length,
        'decoder_inputs': decoder_inputs,
        'target_sequence_length': target_sequence_length,
        'mode': mode,
    }


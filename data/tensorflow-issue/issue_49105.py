from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
import tensorflow_addons as tfa

from las.ops import lstm_cell
from las.ops import pyramidal_bilstm
# assert tf.executing_eagerly()
__all__ = [
    'listener',
    'speller',
]


"""Reference: https://github.com/tensorflow/nmt/blob/master/nmt/gnmt_model.py"""


class AttentionMultiCell(tf.keras.layers.StackedRNNCells):
# class AttentionMultiCell(tf.compat.v1.nn.rnn_cell.MultiRNNCell):
    """A MultiCell with attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=False):
        """Creates a AttentionMultiCell.
        Args:
          attention_cell: An instance of AttentionWrapper.
          cells: A list of RNNCell wrapped with AttentionInputWrapper.
          use_new_attention: Whether to use the attention generated from current
            step bottom layer's output. Default is False.
        """
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(AttentionMultiCell, self).__init__(
            cells)

    def __call__(self, inputs, state, training=False, scope=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.compat.v1.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.compat.v1.variable_scope("cell_0_attention"):
                attention_cell = self.cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(
                    inputs, attention_state)

                new_states.append(new_attention_state)

            for i in range(1, len(self.cells)):
                with tf.compat.v1.variable_scope("cell_%d" % i):

                    cell = self.cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat(
                            [cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat(
                            [cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
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

        super(CustomAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=scale,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name)

        self._query_layer = tf.compat.v1.layers.Dense(
            num_units, name='query_layer', use_bias=False, dtype=dtype)

        self._keys = tf.nn.relu(self.keys)

    def __call__(self, query, state):
        processed_query = tf.nn.relu(self.query_layer(query))

        return super(CustomAttention, self).__call__(processed_query, state)


def listener(encoder_inputs,
             source_sequence_length,
             mode,
             hparams):

    if hparams['use_pyramidal']:
        return pyramidal_bilstm(encoder_inputs, source_sequence_length, mode, hparams)
    else:
        forward_cell_list, backward_cell_list = [], []
        for layer in range(hparams['num_layers']):
            with tf.compat.v1.variable_scope('fw_cell_{}'.format(layer)):
                cell = lstm_cell(hparams['num_units'], hparams['dropout'], mode)

            forward_cell_list.append(cell)

            with tf.compat.v1.variable_scope('bw_cell_{}'.format(layer)):
                cell = lstm_cell(hparams['num_units'], hparams['dropout'], mode)

            backward_cell_list.append(cell)

        forward_cell = tf.keras.layers.StackedRNNCells(forward_cell_list)
        backward_cell = tf.keras.layers.StackedRNNCells(backward_cell_list)

        encoder_outputs, encoder_state = tf.keras.layers.Bidirectional(
            forward_cell,
            backward_cell,
            encoder_inputs,
            sequence_length=source_sequence_length,
            dtype=tf.float32)
        # outputs:[batch_size, max_time, forward_cell.output_size]
        # [batch_size, max_time, hidden_size]

        encoder_outputs = tf.concat(encoder_outputs, -1)

        return (encoder_outputs, source_sequence_length), encoder_state


def attend(encoder_outputs,
           source_sequence_length,
           mode,
           hparams):

    memory = encoder_outputs

    if hparams['attention_type'] == 'luong':
        attention_fn = tfa.seq2seq.LuongAttention
    elif hparams['attention_type'] == 'bahdanau':
        attention_fn = tfa.seq2seq.BahdanauAttention
    elif hparams['attention_type'] == 'custom':
        attention_fn = CustomAttention

    attention_mechanism = attention_fn(
        hparams['num_units'], memory, source_sequence_length)

    cell_list = []
    for layer in range(hparams['num_layers']):

        with tf.compat.v1.variable_scope('decoder_cell_'.format(layer)):
            cell = lstm_cell(hparams['num_units'], hparams['dropout'], mode)

        # cell = lstm_cell(hparams['num_units'], hparams['dropout'], mode)
        cell_list.append(cell)

    alignment_history = (mode != tf.estimator.ModeKeys.TRAIN)

    if hparams['bottom_only']: # False
        #  Only wrap the bottom layer with the attention mechanism.

        attention_cell = cell_list.pop(0)
        # attention_cell = tf.cast(attention_cell, dtype='float32')
        # attention_mechanism = tf.cast(attention_mechanism, dtype='float32')
        attention_cell = tfa.seq2seq.AttentionWrapper(
            attention_cell, attention_mechanism,
            attention_layer_size=hparams['attention_layer_size'],
            alignment_history=alignment_history)

        decoder_cell = AttentionMultiCell(attention_cell, cell_list)
    else:
        decoder_cell = tf.keras.layers.StackedRNNCells(cell_list)

        decoder_cell = tfa.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=hparams['attention_layer_size'],
            alignment_history=alignment_history)

    return decoder_cell


def speller(encoder_outputs,
            encoder_state,
            decoder_inputs,
            source_sequence_length,
            target_sequence_length,
            mode,
            hparams):

    batch_size = tf.shape(input=encoder_outputs)[0]
    beam_width = hparams['beam_width']

    if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        encoder_outputs = tfa.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        source_sequence_length = tfa.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tfa.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = batch_size * beam_width


    if mode == tf.estimator.ModeKeys.EVAL and beam_width > 0:
        encoder_outputs = tfa.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        source_sequence_length = tfa.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tfa.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = batch_size * beam_width

    def embedding_fn(ids):
        # pass callable object to avoid OOM when using one-hot encoding
        if hparams['embedding_size'] != 0:
            target_embedding = tf.compat.v1.get_variable(
                'target_embedding', [
                    hparams['target_vocab_size'], hparams['embedding_size']],
                dtype=tf.float32, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

            return tf.nn.embedding_lookup(params=target_embedding, ids=ids)
        else:
            return tf.one_hot(ids, hparams['target_vocab_size'])

    decoder_cell = attend(
        encoder_outputs, source_sequence_length, mode, hparams)

    projection_layer = tf.keras.layers.Dense(
        hparams['target_vocab_size'], use_bias=True, name='projection_layer')

    if hparams['pass_hidden_state'] and hparams['bottom_only']:
        initial_state = tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tfa.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(
                decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32), encoder_state))
    else:
        initial_state = decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    maximum_iterations = None
    if mode != tf.estimator.ModeKeys.TRAIN:
        max_source_length = tf.reduce_max(input_tensor=source_sequence_length)
        maximum_iterations = tf.cast(tf.round(tf.cast(
            max_source_length, dtype=tf.float32) * hparams['decoding_length_factor']), dtype=tf.int32)

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_inputs = embedding_fn(decoder_inputs)
        decay_steps = hparams['decay_steps']
        iter_num = tf.compat.v1.train.get_global_step()
        inverse_probability = tf.compat.v1.train.polynomial_decay(
            1.0, iter_num, decay_steps, 0.6)
        sampling_probability = 1.0 - inverse_probability
        if hparams['sampling_probability']:
            helper = tfa.seq2seq.ScheduledEmbeddingTrainingSampler(
                sampling_probability=sampling_probability,
                embedding_fn=embedding_fn
            )
        else:
            helper = tfa.seq2seq.TrainingSampler()

        decoder = tfa.seq2seq.BasicDecoder(
            cell=decoder_cell,
            sampler=helper,
            output_layer=projection_layer,
            maximum_iterations=maximum_iterations
        )
     
        decoder_outputs, final_context_state, final_sequence_length = tfa.seq2seq.dynamic_decode(
            decoder, training=True, decoder_init_input=decoder_inputs, decoder_init_kwargs={
                'initial_state': initial_state, 'sequence_length': target_sequence_length
            })

    elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        start_tokens = tf.fill(
            [tf.compat.v1.div(batch_size, beam_width)], hparams['sos_id'])

        decoder = tfa.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding_fn=embedding_fn,
            beam_width=beam_width,
            output_layer=projection_layer,
            maximum_iterations=maximum_iterations
        )
        decoder_outputs, final_context_state, final_sequence_length = tfa.seq2seq.dynamic_decode(
            decoder, decoder_inputs=embedding_fn(decoder_inputs),
            training=False, decoder_init_kwargs={
                'start_tokens': start_tokens, 'end_token': hparams['eos_id'],
                'initial_state': initial_state
            })
    else:
        '''
        start_tokens = tf.fill([batch_size], hparams.sos_id)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_fn, start_tokens, hparams.eos_id)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)
        '''

        start_tokens = tf.fill(
            [tf.compat.v1.div(batch_size, beam_width)], hparams['sos_id'])

        decoder = tfa.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding_fn=embedding_fn,
            beam_width=beam_width,
            output_layer=projection_layer,
            maximum_iterations=maximum_iterations
        )

       

        decoder_outputs, final_context_state, final_sequence_length = tfa.seq2seq.dynamic_decode(
            decoder, decoder_inputs=embedding_fn(decoder_inputs),
            training=False, decoder_init_kwargs={
                'start_tokens':start_tokens,
                'end_token':hparams['eos_id'],
                'initial_state': initial_state
            })

    return decoder_outputs, final_context_state, final_sequence_length
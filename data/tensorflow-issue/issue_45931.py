# tf.random.uniform((B, T), dtype=tf.int32) ← Input shape inferred as (batch_size, time_steps) for token ids and positional ids

import tensorflow as tf
import numpy as np

class TokenWeightingLayer(tf.keras.layers.Layer):
    def __init__(self, mask_constant=-50000, **kwargs):
        super(TokenWeightingLayer, self).__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self._mask_constant = mask_constant

    def call(self, inputs, mask, **kwargs):
        # inputs shape: (batch_size, time_steps, feature_dim)
        activation = tf.squeeze(self.fc(tf.tanh(inputs)), axis=-1)  # shape (batch_size, time_steps)
        # mask: boolean mask tensor shape (batch_size, time_steps)
        masked_activation = tf.where(mask,
                                    activation,
                                    tf.ones_like(activation, dtype=activation.dtype) * self._mask_constant)
        alpha = self.softmax(masked_activation)  # attention weights across time_steps
        return alpha


class MyModel(tf.keras.Model):
    def __init__(self,
                 word_embedding_dim=64,
                 pos1_embedding_dim=5,
                 pos2_embedding_dim=5,
                 vocab_size=100,
                 pos_embedding_num=6,
                 rnn_dim=20,
                 dropout_rate=0.,
                 pre_trained_word_embedding=None,
                 fine_tune_word_embedding=False,
                 variational_recurrent=False,
                 num_cls=10,
                 **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Embeddings
        self._word_embedding_dim = word_embedding_dim
        self._entity_1_embedding_dim = pos1_embedding_dim
        self._entity_2_embedding_dim = pos2_embedding_dim

        self._vocab_size = vocab_size
        self._pos_embedding_num = pos_embedding_num
        self._rnn_dim = rnn_dim
        self._num_cls = num_cls
        self._dropout_rate = dropout_rate

        if pre_trained_word_embedding is None:
            self.token_embedding = tf.keras.layers.Embedding(self._vocab_size, self._word_embedding_dim, mask_zero=True)
        else:
            # Not implemented: loading pretrained embeddings with fine-tuning option
            raise NotImplementedError("Pretrained embeddings not supported in this implementation.")

        # Use shared positional embedding for pos1 and pos2 as in original code
        self._pos_embedding_1 = tf.keras.layers.Embedding(self._pos_embedding_num, self._word_embedding_dim, mask_zero=True)
        self._pos_embedding_2 = self._pos_embedding_1

        # BiDirectional RNN setup with GRUCell wrapped into RNN layer and Bidirectional wrapper
        cell = tf.keras.layers.GRUCell(self._rnn_dim)
        rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
        self.rnn_encoder = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")

        self.token_weighting_layer = TokenWeightingLayer()
        self.fc_layer = tf.keras.layers.Dense(self._num_cls, use_bias=True)

    def call(self, token_ids, pos_ids_1, pos_ids_2, sequence_length, training=None, mask=None):
        """
        token_ids: tf.int32 tensor (batch_size, time_steps)
        pos_ids_1: tf.int32 tensor (batch_size, time_steps)
        pos_ids_2: tf.int32 tensor (batch_size, time_steps)
        sequence_length: tf.int32 tensor (batch_size,)
        """
        token_embedding = self.token_embedding(token_ids)  # shape (batch, time_steps, word_embedding_dim)
        pos_embedding_1 = self._pos_embedding_1(pos_ids_1)  # shape (batch, time_steps, word_embedding_dim)
        pos_embedding_2 = self._pos_embedding_2(pos_ids_2)  # same shape

        # Concatenate embeddings along last dim
        encoded_input = tf.concat([token_embedding, pos_embedding_1, pos_embedding_2], axis=-1)  # (batch, time_steps, dim)

        # Compute mask based on sequence_length
        mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(token_ids)[1])

        # Bidirectional RNN Encoder: outputs shape (batch, time_steps, rnn_dim), states are final hidden states
        encoder_outputs, last_state_fw, last_state_bw = self.rnn_encoder(encoded_input, mask=mask, training=training)

        # Attention weights over time steps
        alpha = self.token_weighting_layer(encoder_outputs, mask)

        # Weighted sum over encoder_outputs by alpha: shape (batch, rnn_dim)
        weighted_outputs = tf.squeeze(tf.matmul(tf.expand_dims(alpha, axis=1), encoder_outputs), axis=1)

        # Final classification layer logits
        out = self.fc_layer(weighted_outputs)  # (batch, num_cls)
        return out, alpha

    def get_config(self):
        # Enable serialization, minimal config
        config = super(MyModel, self).get_config()
        config.update({
            "word_embedding_dim": self._word_embedding_dim,
            "pos1_embedding_dim": self._entity_1_embedding_dim,
            "pos2_embedding_dim": self._entity_2_embedding_dim,
            "vocab_size": self._vocab_size,
            "pos_embedding_num": self._pos_embedding_num,
            "rnn_dim": self._rnn_dim,
            "dropout_rate": self._dropout_rate,
            "num_cls": self._num_cls,
        })
        return config

def my_model_function():
    # Return an instance of MyModel with default parameters as per the issue's example
    model = MyModel(word_embedding_dim=64,
                    pos1_embedding_dim=5,
                    pos2_embedding_dim=5,
                    pos_embedding_num=6,
                    vocab_size=100,
                    rnn_dim=20,
                    dropout_rate=0.0,
                    pre_trained_word_embedding=None,
                    fine_tune_word_embedding=False,
                    variational_recurrent=False,
                    num_cls=10)
    return model

def GetInput():
    # Return a random tensor input tuple matching MyModel expected inputs:
    # token_ids: (batch_size, time_steps) int32
    # pos_ids_1: (batch_size, time_steps) int32
    # pos_ids_2: (batch_size, time_steps) int32
    # sequence_length: (batch_size,) int32

    batch_size = 3
    time_steps = 8
    vocab_size = 100
    pos_embedding_num = 6

    # Generate random token ids in valid vocab range [1, vocab_size-1], zero for padding
    # To simulate variable lengths, randomly zero out trailing tokens and set sequence_length accordingly
    token_ids_np = np.array([
        [1, 2, 3, 4, 5, 0, 0, 0],
        [1, 2, 3, 4, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8]
    ], dtype=np.int32)

    pos_ids_1_np = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0]
    ], dtype=np.int32)

    pos_ids_2_np = pos_ids_1_np.copy()  # As per model, same embedding shared

    sequence_length_np = np.array([5, 4, 8], dtype=np.int32)

    token_ids = tf.constant(token_ids_np)
    pos_ids_1 = tf.constant(pos_ids_1_np)
    pos_ids_2 = tf.constant(pos_ids_2_np)
    sequence_length = tf.constant(sequence_length_np)

    return (token_ids, pos_ids_1, pos_ids_2, sequence_length)


# tf.random.uniform((B, None), dtype=tf.int32) ← Input is a batch of variable-length sequences of token indices (integers)

import tensorflow as tf
import numpy as np


def generate_square_subsequent_mask(size):
    """Generate a square mask for the sequence.
    Masked positions are filled with 1.0, unmasked with 0.0."""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_multihead_mask(x, x_length, y, reverse=False):
    """Generate masks for multi-head attention.
    Masks are float tensors with 1.0 for masked positions and 0.0 for others.
    x: input tensor (not used here beyond shape)
    x_length: tensor of lengths for x (shape: batch)
    y: target tensor (optional)
    reverse: controls look-ahead behavior
    """
    x_mask, y_mask = None, None
    if x is not None:
        x_mask = 1.0 - tf.sequence_mask(
            x_length, tf.shape(x)[1], dtype=tf.float32
        )
        x_mask = tf.expand_dims(tf.expand_dims(x_mask, 1), 1)
        if reverse:
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(x)[1])
            x_mask = tf.maximum(x_mask, look_ahead_mask)
        x_mask.set_shape([None, None, None, None])
    if y is not None:
        y_mask = tf.cast(tf.math.equal(y, 0), tf.float32)
        y_mask = tf.expand_dims(tf.expand_dims(y_mask, 1), 1)
        if not reverse:
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(y)[1])
            y_mask = tf.maximum(y_mask, look_ahead_mask)
        y_mask.set_shape([None, None, None, None])
    return x_mask, y_mask


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer models."""

    def make_positional_encoding(self, position, d_model):
        """Generate positional encoding as per Vaswani et al."""
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(
            np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(self, d_model, max_position=800, scale=False):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.pos_encoding = self.make_positional_encoding(max_position, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        if self.scale:
            x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        return x


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding with a trainable scaling parameter alpha.
    See https://arxiv.org/pdf/1809.08895.pdf."""

    def __init__(self, d_model, max_position=800):
        super().__init__(d_model, max_position, scale=False)

    def build(self, _):
        self.alpha = self.add_weight(
            name="alpha", initializer=tf.keras.initializers.constant(1), trainable=True
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.alpha * self.pos_encoding[:, :seq_len, :]
        return x


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Calculate scaled dot-product attention."""

    def __init__(self, unidirectional=False, look_ahead=0):
        super().__init__()
        self.uni = unidirectional
        self.look_ahead = look_ahead

    def call(self, q, k, v, mask):
        # q: (..., seq_len_q, depth)
        # k: (..., seq_len_k, depth)
        # v: (..., seq_len_v, depth_v)
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if self.uni:
            uni_mask = tf.ones(tf.shape(scaled_attention_logits))
            uni_mask = tf.linalg.band_part(uni_mask, -1, self.look_ahead)
            scaled_attention_logits += (1 - uni_mask) * -1e9
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head self-attention as in the transformer architecture."""

    def __init__(self, d_model, num_heads, unidirectional=False, look_ahead=0):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        dense_init = tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=dense_init, input_shape=(d_model,))
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=dense_init, input_shape=(d_model,))
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=dense_init, input_shape=(d_model,))

        self.attention = ScaledDotProductAttention(unidirectional, look_ahead=look_ahead)
        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=dense_init, input_shape=(d_model,))

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Single transformer encoder layer."""

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
        unidirectional=False, look_ahead=0, ffn=None
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, unidirectional, look_ahead=look_ahead)

        layers = tf.keras.layers
        if ffn is None:
            self.ffn = tf.keras.Sequential([
                layers.Dense(dim_feedforward, activation='gelu',
                             kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                             input_shape=(d_model,)),
                layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                layers.Dense(d_model, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                             input_shape=(dim_feedforward,)),
                layers.Dropout(dropout, input_shape=(d_model,))
            ])
        else:
            self.ffn = ffn

        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout = layers.Dropout(dropout, input_shape=(d_model,))

    def call(self, src, src_mask=None, training=None):
        out = self.self_attn(src, src, src, mask=src_mask)[0]
        out = self.norm1(src + self.dropout(out, training=training))
        out = self.norm2(out + self.ffn(out, training=training))
        return out


class TransformerEncoder(tf.keras.layers.Layer):
    """Stack of transformer encoder layers."""

    def __init__(self, encoder_layers):
        super().__init__()
        self.layers = encoder_layers

    def call(self, src, src_mask=None, training=None):
        output = src
        for i in range(len(self.layers)):
            output = self.layers[i](output, src_mask=src_mask, training=training)
        return output


class MyModel(tf.keras.Model):
    """
    Implementing the FastSpeech Transformer-based TTS model as given in the issue,
    wrapped as `MyModel`.
    """

    def __init__(self):
        super().__init__()

        self.num_class = 314     # vocabulary size (token classes)
        self.eos = self.num_class - 1
        self.feat_dim = 80      # output mel-spectrogram feature dimension
        self.reduction_factor = 1

        layers = tf.keras.layers

        # x_net: embedding + scaled positional encoding + dropout
        input_features = layers.Input(shape=(None,), dtype=tf.int32)
        inner = layers.Embedding(self.num_class, 384)(input_features)  # embedding dim = 384
        inner = ScaledPositionalEncoding(384)(inner)
        inner = layers.Dropout(0.1)(inner)
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")

        # ffn_list: two FFN blocks implemented as Conv1D + ReLU + Dropout + Conv1D
        ffn_list = []
        for _ in range(2):
            ffn_list.append(tf.keras.Sequential([
                layers.Conv1D(filters=1536, kernel_size=3, strides=1, padding="same", use_bias=False, data_format="channels_last"),
                layers.ReLU(),
                layers.Dropout(0.1),
                layers.Conv1D(filters=384, kernel_size=3, strides=1, padding="same", use_bias=False, data_format="channels_last"),
            ]))

        # Encoder: 6 TransformerEncoderLayer blocks with FFN ffn_list[0]
        encoder_layers = [TransformerEncoderLayer(384, 2, 1536, 0.1, ffn=ffn_list[0]) for _ in range(6)]
        self.encoder = TransformerEncoder(encoder_layers)

        # Duration predictor: 2 Conv1D layers + ReLU + LayerNorm + Dropout + Dense(1)
        input_features = layers.Input(shape=(None, 384), dtype=tf.float32)
        inner = input_features
        for _ in range(2):
            inner = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding="same",
                                  use_bias=False, data_format="channels_last")(inner)
            inner = layers.ReLU()(inner)
            inner = layers.LayerNormalization()(inner)
            inner = layers.Dropout(0.1)(inner)
        inner = layers.Dense(1)(inner)  # output shape: [batch, length, 1]
        inner = tf.squeeze(inner, axis=-1)  # [batch, length]
        self.duration_predictor = tf.keras.Model(inputs=input_features, outputs=inner, name="duration_predictor")

        # y_net: scaled positional encoding + dropout applied to encoder output
        input_features = layers.Input(shape=(None, 384), dtype=tf.float32)
        inner = ScaledPositionalEncoding(384, max_position=3200)(input_features)
        inner = layers.Dropout(0.1)(inner)
        self.y_net = tf.keras.Model(inputs=input_features, outputs=inner, name="y_net")

        # Decoder: 6 TransformerEncoderLayer blocks with FFN ffn_list[1]
        decoder_layers = [TransformerEncoderLayer(384, 2, 1536, 0.1, ffn=ffn_list[1]) for _ in range(6)]
        self.decoder = TransformerEncoder(decoder_layers)

        # feat_out projection to mel spectrogram dimension
        self.feat_out = layers.Dense(self.feat_dim * self.reduction_factor, use_bias=False, name='feat_projection')

        # postnet: five Conv1D + BatchNorm + tanh + Dropout layers followed by Dense projection
        input_features_postnet = layers.Input(shape=(None, self.feat_dim), dtype=tf.float32)
        inner = input_features_postnet
        for _ in range(5):
            filters = 256
            inner = layers.Conv1D(filters=filters, kernel_size=5, strides=1, padding="same",
                                  use_bias=False, data_format="channels_last")(inner)
            inner = layers.BatchNormalization()(inner)
            inner = tf.nn.tanh(inner)
            inner = layers.Dropout(0.5)(inner)
        inner = layers.Dense(self.feat_dim, name='projection')(inner)
        self.postnet = tf.keras.Model(inputs=input_features_postnet, outputs=inner, name="postnet")

    def _feedforward_decoder(self, expanded_array, expanded_length, training=None):
        """Decoder feed-forward pass after length regulation.

        Args:
          expanded_array: tensor of shape [batch, expanded_length, d_model]
          expanded_length: tensor of shape [batch]
          training: boolean flag for training

        Returns:
          before_outs: tensor before postnet correction [batch, expanded_length, feat_dim]
          after_outs: tensor after postnet correction [batch, expanded_length, feat_dim]
        """
        expanded_mask, _ = create_multihead_mask(expanded_array, expanded_length, None)
        expanded_output = self.y_net(expanded_array, training=training)
        decoder_output = self.decoder(expanded_output, expanded_mask, training=training)
        batch = tf.shape(decoder_output)[0]
        decoder_output = self.feat_out(decoder_output, training=training)
        before_outs = tf.reshape(decoder_output, [batch, -1, self.feat_dim])
        after_outs = before_outs + self.postnet(before_outs, training=training)
        return before_outs, after_outs

    def call(self, samples, training=None):
        """
        The forward call is not fully implemented in the original code snippet for training.
        Returns None to stay compatible.
        """
        return None

    def synthesize(self, samples, alpha=1.0):
        """
        FastSpeech inference path producing mel spectrogram outputs.

        Args:
          samples: dict with key 'input', tensor of int32 shape [batch, seq_len]
          alpha: controls speech speed (not used here but could scale durations)

        Returns:
          mel spectrogram tensor after postnet: [batch, time, feat_dim]
        """
        x0 = self.x_net(samples['input'], training=False)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=False)  # [batch, x_steps, d_model]

        duration_pred = self.duration_predictor(encoder_output, training=False)
        duration_pred = tf.clip_by_value(
            tf.math.round(tf.exp(duration_pred) - 1.0),
            0.0, 100.0)
        duration_sequences = tf.cast(duration_pred, tf.int32)  # [batch, x_steps]

        # Length regulation: repeat encoder outputs per duration sequence
        batch = tf.shape(encoder_output)[0]
        phoneme_seq = encoder_output[0]  # [x_step, d_model], taking batch=1 for synthesis
        duration_seq = duration_sequences[0]  # [x_step]

        repeated_phoneme_seq = tf.repeat(phoneme_seq, repeats=duration_seq, axis=0)
        repeated_phoneme_seq = tf.expand_dims(repeated_phoneme_seq, axis=0)  # Add batch dim=1

        expanded_length = tf.reduce_sum(duration_sequences, axis=1)  # [batch]

        _, after_outs = self._feedforward_decoder(repeated_phoneme_seq, expanded_length, training=False)
        return after_outs


def my_model_function():
    # Return an instance of MyModel with default initialization.
    return MyModel()


def GetInput():
    # Input is a batch of integer sequences with batch size=1 and variable length.
    # Using a sample sequence of token indices within [0,313]
    sample_seq = [312, 223, 131, 117, 66, 200, 233, 199, 217, 308, 278, 309, 248, 312, 313]
    tensor_input = tf.constant([sample_seq], dtype=tf.int32)
    return tensor_input


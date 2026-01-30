import math
import random
from tensorflow import keras
from tensorflow.keras import layers

interpreter = tf.lite.Interpreter(model_path="model.tflite")
for samples in data_queue:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.resize_tensor_input(input_details[0]['index'], samples["input"].shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], samples["input"])
            interpreter.invoke()
            features = interpreter.get_tensor(output_details[0]['index'])
            self.vocoder(features.numpy())

for samples in data_queue:
            interpreter = tf.lite.Interpreter(model_path="model.tflite")
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.resize_tensor_input(input_details[0]['index'], samples["input"].shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], samples["input"])
            interpreter.invoke()
            features = interpreter.get_tensor(output_details[0]['index'])
            self.vocoder(features.numpy())

import tensorflow as tf
import numpy as np


def generate_square_subsequent_mask(size):
    """  Generate a square mask for the sequence. The masked positions are filled with float(1.0).
      Unmasked positions are filled with float(0.0).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_multihead_mask(x, x_length, y, reverse=False):
    r""" Generate a square mask for the sequence for mult-head attention.
        The masked positions are filled with float(1.0).
        Unmasked positions are filled with float(0.0).
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
    """ positional encoding can be used in transformer """

    def make_positional_encoding(self, position, d_model):
        """ generate a postional encoding list """

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
        """ call function """
        seq_len = tf.shape(x)[1]
        if self.scale:
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        return x


class ScaledPositionalEncoding(PositionalEncoding):
    """ scaled positional encoding,
        reference: https://arxiv.org/pdf/1809.08895.pdf"""
    def __init__(self, d_model, max_position=800):
        super().__init__(d_model, max_position, scale=False)

    def build(self, _):
        self.alpha = self.add_weight(
            name="alpha", initializer=tf.keras.initializers.constant(1)
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.alpha * self.pos_encoding[:, :seq_len, :]
        return x


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    def __init__(self, unidirectional=False, look_ahead=0):
        super().__init__()
        self.uni = unidirectional
        self.look_ahead = look_ahead

    def call(self, q, k, v, mask):
        """This is where the layer's logic lives."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if self.uni:
            uni_mask = tf.ones(tf.shape(scaled_attention_logits))
            uni_mask = tf.linalg.band_part(uni_mask, -1, self.look_ahead)
            scaled_attention_logits += (1 - uni_mask) * -1e9
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi-head attention

    Multi-head attention consists of four parts: * Linear layers and split into
    heads. * Scaled dot-product attention. * Concatenation of heads. * Final linear layer.
    Each multi-head attention block gets three inputs; Q (query), K (key), V (value).
    These are put through linear (Dense) layers and split up into multiple heads.
    The scaled_dot_product_attention defined above is applied to each head (broadcasted for
    efficiency). An appropriate mask must be used in the attention step. The attention
    output for each head is then concatenated (using tf.transpose, and tf.reshape) and
    put through a final Dense layer.
    Instead of one single attention head, Q, K, and V are split into multiple heads because
    it allows the model to jointly attend to information at different positions from
    different representational spaces. After the split each head has a reduced dimensionality,
    so the total computation cost is the same as a single head attention with full
    dimensionality.
    """

    def __init__(self, d_model, num_heads, unidirectional=False, look_ahead=0):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )
        self.wk = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )
        self.wv = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )

        self.attention = ScaledDotProductAttention(unidirectional, look_ahead=look_ahead)

        self.dense = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """ call function """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, hiddn_dim)
        k = self.wk(k)  # (batch_size, seq_len, hiddn_dim)
        v = self.wv(v)  # (batch_size, seq_len, hiddn_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = tf.random(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
            unidirectional=False, look_ahead=0, ffn=None
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, unidirectional, look_ahead=look_ahead)
        # Implementation of Feedforward model
        layers = tf.keras.layers
        if ffn is None:
            self.ffn = tf.keras.Sequential(
                [
                    layers.Dense(
                        dim_feedforward,
                        activation='gelu',
                        kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                            stddev=0.02
                        ),
                        input_shape=(d_model,),
                    ),
                    layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                    layers.Dense(
                        d_model,
                        kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                            stddev=0.02
                        ),
                        input_shape=(dim_feedforward,),
                    ),
                    layers.Dropout(dropout, input_shape=(d_model,)),
                ]
            )
        else:
            self.ffn = ffn

        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout = layers.Dropout(dropout, input_shape=(d_model,))

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        out = self.self_attn(src, src, src, mask=src_mask)[0]
        out = self.norm1(src + self.dropout(out, training=training))
        out = self.norm2(out + self.ffn(out, training=training))

        return out


class TransformerEncoder(tf.keras.layers.Layer):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = [TransformerEncoderLayer(d_model=512, nhead=8)
        >>>                    for _ in range(num_layers)]
        >>> transformer_encoder = TransformerEncoder(encoder_layer)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layers):
        super().__init__()
        self.layers = encoder_layers

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        for i in range(len(self.layers)):
            output = self.layers[i](output, src_mask=src_mask, training=training)
        return output


class FastSpeech(tf.keras.Model):
    """
    Reference: Fastspeech: Fast, robust and controllable text to speech
      (http://papers.nips.cc/paper/8580-fastspeech-fast-robust-and-controllable-text-to-speech.pdf)
    """

    def __init__(self):
        super().__init__()

        self.num_class = 314
        self.eos = self.num_class - 1
        self.feat_dim = 80
        self.reduction_factor = 1

        # for the x_net
        layers = tf.keras.layers
        input_features = layers.Input(shape=tf.TensorShape([None]), dtype=tf.int32)
        inner = layers.Embedding(self.num_class, 384)(input_features)
        inner = ScaledPositionalEncoding(384)(inner)
        inner = layers.Dropout(0.1)(inner)
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        ffn_list = []
        for _ in range(2):
            ffn_list.append(tf.keras.Sequential(
                [
                    layers.Conv1D(
                        filters=1536,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format="channels_last"),
                    layers.ReLU(),
                    layers.Dropout(0.1),
                    layers.Conv1D(
                        filters=384,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format="channels_last")
                ]
            ))
        # define encoder form transform.py
        encoder_layers = [
            TransformerEncoderLayer(384, 2, 1536, 0.1, ffn=ffn_list[0])
            for _ in range(6)
        ]
        self.encoder = TransformerEncoder(encoder_layers)

        # define duration predictor
        input_features = layers.Input(shape=tf.TensorShape([None, 384]),
                                       dtype=tf.float32)
        inner = input_features
        for _ in range(2):
            inner = layers.Conv1D(
                filters=256,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias = False,
                data_format = "channels_last")(inner)
            inner = layers.ReLU()(inner)
            inner = layers.LayerNormalization()(inner)
            inner = layers.Dropout(0.1)(inner)
        inner = layers.Dense(1)(inner) # [batch, expanded_length, 1]
        inner = tf.squeeze(inner, axis=-1)
        self.duration_predictor = tf.keras.Model(inputs=input_features, outputs=inner,
                                                 name="duration_predictor")
        print(self.duration_predictor.summary())

        # for the y_net
        input_features = layers.Input(shape=tf.TensorShape([None, 384]),
                                      dtype=tf.float32)
        inner = ScaledPositionalEncoding(384, max_position=3200)(input_features)
        inner = layers.Dropout(0.1)(inner)
        self.y_net = tf.keras.Model(inputs=input_features, outputs=inner, name="y_net")
        print(self.y_net.summary())

        # define decoder
        decoder_layers = [
            TransformerEncoderLayer(384, 2, 1536, 0.1, ffn=ffn_list[1])
            for _ in range(6)
        ]
        self.decoder = TransformerEncoder(decoder_layers)

        # define feat_out
        self.feat_out = layers.Dense(self.feat_dim * self.reduction_factor, use_bias=False,
                                     name='feat_projection')
        # define postnet
        input_features_postnet = layers.Input(shape=tf.TensorShape([None, self.feat_dim]),
                                              dtype=tf.float32)
        inner = input_features_postnet
        for _ in tf.range(5):
            filters = 256
            inner = layers.Conv1D(
                filters=filters,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                data_format="channels_last",
            )(inner)
            inner = layers.BatchNormalization()(inner)
            inner = tf.nn.tanh(inner)
            inner = layers.Dropout(0.5)(inner)
        inner = layers.Dense(self.feat_dim, name='projection')(inner)
        self.postnet = tf.keras.Model(inputs=input_features_postnet, outputs=inner, name="postnet")
        print(self.postnet.summary())

    def get_loss(self, outputs, samples, training=None):
        """ get loss used for training """
        return None, None

    def _feedforward_decoder(self, expanded_array, expanded_length, training: bool = None):
        """feed-forward decoder
        Args:
            expanded_array: expanded encoder outputs after length regulation
                shape: [batch, y_steps, d_model]
            expanded_length: corresponding lengths, shape: [batch, y_steps]
            training: if it is in the training stage
        Returns:
            before_outs: the outputs before postnet calculation
            after_outs: the outputs after postnet calculation
        """

        expanded_mask, _ = create_multihead_mask(expanded_array, expanded_length, None)
        expanded_output = self.y_net(expanded_array, training=training)
        # decoder_output, shape: [batch, expanded_length, d_model]
        decoder_output = self.decoder(expanded_output, expanded_mask, training=training)
        batch = tf.shape(decoder_output)[0]
        decoder_output = self.feat_out(decoder_output, training=training)
        before_outs = tf.reshape(decoder_output, [batch, -1, self.feat_dim])
        after_outs = before_outs + self.postnet(before_outs, training=training)
        return before_outs, after_outs

    def call(self, samples, training: bool = None):
        return None

    def synthesize(self, samples, alpha=1.0):
        x0 = self.x_net(samples['input'], training=False)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=False) # [batch, x_steps, d_model]
        duration_sequences = self.duration_predictor(encoder_output, training=False)
        duration_sequences = tf.cast(tf.clip_by_value(tf.math.round(
            tf.exp(duration_sequences) - 1.0),
            0.0, tf.cast(100, dtype=tf.float32)), dtype=tf.int32)

        phoneme_seq = encoder_output[0]  # [x_step, d_model]
        duration_seq = duration_sequences[0]  # [x_step]
        repeated_phoneme_seq = tf.repeat(phoneme_seq, repeats=duration_seq, axis=0)
        repeated_phoneme_seq = tf.expand_dims(repeated_phoneme_seq, axis=0)
        expanded_length = tf.reduce_sum(duration_sequences, axis=1)  # [batch]

        _, after_outs = self._feedforward_decoder(repeated_phoneme_seq, expanded_length, training=False)
        return after_outs


def tflite_convert():
    """ restore the best model """
    model = FastSpeech()
    checkpointer = tf.train.Checkpoint(model=model)
    checkpointer.restore(tf.train.latest_checkpoint("tmp_ckpt/"))

    def inference(x):
        samples = {"input": x}
        outputs = model.synthesize(samples)
        return outputs

    model.inference_function = tf.function(inference, experimental_relax_shapes=True,
                                           input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.int32)])
    tf.saved_model.save(obj=model, export_dir="tmp_model")
    load_model = tf.saved_model.load("tmp_model")

    concrete_function = load_model.inference_function.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    tf.random.set_seed(1)
    tflite_convert()

def _feedforward_decoder(self, expanded_array, expanded_length, training: bool = None):
        seq_len = tf.shape(expanded_array)[1]
        expanded_output = expanded_array + 0.69105166 * self.position[:, :seq_len, :]
        expanded_output = tf.concat([expanded_output, self.position[:, :seq_len, :]], axis=0)
        return None, expanded_output
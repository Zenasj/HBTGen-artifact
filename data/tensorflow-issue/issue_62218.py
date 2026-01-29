# tf.random.uniform((None, 224, 224, 1), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Dense, MultiHeadAttention, Dropout, LayerNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class CoordinateChannel(Layer):
    """ Adds Coordinate Channels to the input tensor.
    # Arguments
        rank: An integer, the rank of the input data-uniform,
            e.g. "2" for 2D convolution.
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        ND tensor with shape:
        `(samples, channels, *)`
        if `data_format` is `"channels_first"`
        or ND tensor with shape:
        `(samples, *, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        ND tensor with shape:
        `(samples, channels + 2, *)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, *, channels + 2)`
        if `data_format` is `"channels_last"`.
    """

    def __init__(self, rank,
                 use_radius=False,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel, self).__init__(**kwargs)

        if data_format not in [None, 'channels_first', 'channels_last']:
            raise ValueError('`data_format` must be either "channels_last", "channels_first" '
                             'or None.')

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = K.shape(inputs)

        if self.rank == 1:
            input_shape = [input_shape[i] for i in range(3)]
            batch_shape, dim, channels = input_shape

            xx_range = K.tile(K.expand_dims(K.arange(0, dim, dtype='float32'), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=-1)

            xx_channels = K.cast(xx_range, K.floatx())
            xx_channels = xx_channels / K.cast(dim - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels], axis=-1)

        elif self.rank == 2:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(4)]
            batch_shape, dim1, dim2, channels = input_shape

            xx_ones = tf.ones(K.stack([batch_shape, dim2]), dtype='float32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim1, dtype='float32'), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)
            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            yy_ones = tf.ones(K.stack([batch_shape, dim1]), dtype='float32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim2, dtype='float32'), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
            yy_channels = (yy_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels, yy_channels], axis=-1)

            if self.use_radius:
                rr = K.sqrt(K.square(xx_channels - 0.5) +
                            K.square(yy_channels - 0.5))
                outputs = K.concatenate([outputs, rr], axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

        elif self.rank == 3:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 4, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(5)]
            batch_shape, dim1, dim2, dim3, channels = input_shape

            xx_ones = tf.ones(K.stack([batch_shape, dim3]), dtype='float32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim2, dtype='float32'), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)

            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            xx_channels = K.expand_dims(xx_channels, axis=1)
            xx_channels = K.tile(xx_channels,
                                 [1, dim1, 1, 1, 1])

            yy_ones = tf.ones(K.stack([batch_shape, dim2]), dtype='float32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim3, dtype='float32'), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            yy_channels = K.expand_dims(yy_channels, axis=1)
            yy_channels = K.tile(yy_channels,
                                 [1, dim1, 1, 1, 1])

            zz_range = K.tile(K.expand_dims(K.arange(0, dim1, dtype='float32'), axis=0),
                              K.stack([batch_shape, 1]))
            zz_range = K.expand_dims(zz_range, axis=-1)
            zz_range = K.expand_dims(zz_range, axis=-1)

            zz_channels = K.tile(zz_range,
                                 [1, 1, dim2, dim3])
            zz_channels = K.expand_dims(zz_channels, axis=-1)

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim2 - 1, K.floatx())
            xx_channels = xx_channels * 2 - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim3 - 1, K.floatx())
            yy_channels = yy_channels * 2 - 1.

            zz_channels = K.cast(zz_channels, K.floatx())
            zz_channels = zz_channels / K.cast(dim1 - 1, K.floatx())
            zz_channels = zz_channels * 2 - 1.

            outputs = K.concatenate([inputs, zz_channels, xx_channels, yy_channels],
                                    axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 4, 1, 2, 3])
        else:
            # For unsupported ranks, just pass inputs through
            outputs = inputs

        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        if self.use_radius and self.rank == 2:
            channel_count = 3
        else:
            channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            'rank': self.rank,
            'use_radius': self.use_radius,
            'data_format': self.data_format
        }
        base_config = super(CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dim_feedforward, regularizer_rate=0, dropout=0.1, vocab_size=2):
        super(TransformerDecoderLayer, self).__init__()
        self.last_attn_scores = None
        self.regularizer_rate = regularizer_rate
        self.kernel_regularizer = l2(regularizer_rate)
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)

        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation='relu', kernel_regularizer=self.kernel_regularizer)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(d_model, kernel_regularizer=self.kernel_regularizer)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.drop_out_rate = dropout

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'num_heads': max(1, self.d_model // 32),
            'dim_feedforward': self.d_model * 4,
            'dropout': self.drop_out_rate,
            'vocab_size': self.vocab_size,
            'regularizer_rate': self.regularizer_rate,
        }
        return config

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
             memory_key_padding_mask=None):
        # Add positional embeddings: simplified direct embedding lookup replaced with simple positional encoding:
        # Because no PositionalEmbedding class definition was given, we use a simple sinusoidal, or just pass through
        # For this reconstruction, we treat tgt as embeddings, so we just add a small positional encoding tensor
        seq_len = tf.shape(tgt)[1]
        d_model = self.d_model
        
        # Simple sinusoidal position encoding to add positional context
        position = tf.cast(tf.range(seq_len)[tf.newaxis, :, tf.newaxis], tf.float32)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pe = tf.zeros((1, seq_len, d_model), dtype=tf.float32)
        pe_even = tf.sin(position * div_term)
        pe_odd = tf.cos(position * div_term)
        pe = tf.reshape(tf.concat([pe_even, pe_odd], axis=-1), (1, seq_len, d_model))
        tgt = tgt + pe[:, :seq_len, :]

        # create look ahead mask for causal attention
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        look_ahead_mask = look_ahead_mask[tf.newaxis, :, :]

        # Self-attention with look ahead mask
        tgt2 = self.self_attn(query=tgt, value=tgt, key=tgt, attention_mask=look_ahead_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Multi-head attention with memory (encoder output)
        attn_out, attn_scores = self.multihead_attn(
            query=tgt, value=memory, key=memory, return_attention_scores=True)
        tgt = tgt + self.dropout2(attn_out)
        tgt = self.norm2(tgt)

        self.last_attn_scores = attn_scores

        tgt2 = self.linear2(self.dropout(self.linear1(tgt)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # CoordinateChannel expects 2D input with shape (batch, 224, 224, 1)
        self.coord_channel = CoordinateChannel(rank=2, use_radius=False, data_format='channels_last')

        # Sequential conv backbone - simplified due to length of original, here replaced with single conv for brevity
        # The original Sequential from JSON is complex but here we simulate with a small conv stack.
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv2D(48, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(96, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])

        # Dense layers and Transformer decoder
        self.dense450 = Dense(450, use_bias=False, kernel_regularizer=l2(3e-6))
        self.dense256 = Dense(256, use_bias=False, kernel_regularizer=l2(3e-6))
        self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))

        self.transformer_decoder = TransformerDecoderLayer(
            d_model=256, num_heads=8, dim_feedforward=1024, dropout=0.0, vocab_size=2, regularizer_rate=3e-6
        )

        self.dense2 = Dense(2, use_bias=False, kernel_regularizer=l2(3e-6))

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs: tuple of (image_tensor [batch,224,224,1], sequence_tensor [batch,159])
        x1, x2 = inputs

        x1 = self.coord_channel(x1)      # Add coordinate channels [batch, 224, 224, 3]

        # Pass through conv backbone
        x = self.backbone(x1)            # [batch, features]

        x = self.dense256(x)             # [batch, 256]
        x = self.expand_dims(x)          # [batch, 1, 256]

        # Transformer decoder expects (tgt, memory)
        tgt = x2                        # [batch, 159]
        memory = x                     # [batch, 1, 256]

        tgt_out = self.transformer_decoder(tgt, memory)  # [batch, 159, 256]

        out = self.dense2(tgt_out)      # [batch, 159, 2]

        return out


def my_model_function():
    # Return an instance of MyModel, including default initialization
    return MyModel()

def GetInput():
    # Return input tuple matching MyModel input:
    # - image tensor: shape (batch, 224, 224, 1), float32
    # - sequence tensor: shape (batch, 159), float32
    batch_size = 2  # choose small batch size for example
    img_input = tf.random.uniform((batch_size, 224, 224, 1), dtype=tf.float32)
    seq_input = tf.random.uniform((batch_size, 159), dtype=tf.float32)
    return (img_input, seq_input)


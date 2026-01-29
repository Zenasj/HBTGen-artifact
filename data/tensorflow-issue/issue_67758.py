# tf.random.uniform((B, SEQ_LEN, D_MODEL), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense, MultiHeadAttention
from tensorflow.keras import Model

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, t_num_heads, t_key_dim, t_ff_dim, dropout_rate=0.1, activation_function='relu',
                 initializer='glorot_uniform', **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=t_num_heads, key_dim=t_key_dim)
        self.ffn = tf.keras.Sequential([
            Dense(t_ff_dim, activation=activation_function, kernel_initializer=initializer),
            Dense(t_key_dim, kernel_initializer=initializer),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dense_proj = Dense(t_key_dim, kernel_initializer=initializer)

    def call(self, inputs, training=None):
        # Project input to key dimension space
        inputs_proj = self.dense_proj(inputs)
        # Remove singleton dimension if shape is (B, H, 1, C)
        if len(inputs_proj.shape) == 4 and inputs_proj.shape[2] == 1:
            inputs_proj = tf.squeeze(inputs_proj, axis=2)
        attn_output = self.att(inputs_proj, inputs_proj)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs_proj + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            't_num_heads': self.att.num_heads,
            't_key_dim': self.att.key_dim,
            't_ff_dim': self.ffn.layers[0].units,
            'dropout_rate': self.dropout1.rate,
            'activation_function': self.ffn.layers[0].activation.__name__,
            'initializer': self.ffn.layers[0].kernel_initializer.__class__.__name__
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters inferred from example usage in the issue
        self.num_heads = 8
        self.key_dim = 64
        self.d_ff = 256
        self.d_model = 512
        self.n_layers = 6  # stack size
        
        # Positional embedding approximated as learned Embedding for vocab size 10000 and d_model
        self.encoder_vocab_len = 10000
        self.decoder_vocab_len = 10000
        
        # Embeddings
        self.encoder_embedding = tf.keras.layers.Embedding(self.encoder_vocab_len, self.d_model)
        self.decoder_embedding = tf.keras.layers.Embedding(self.decoder_vocab_len, self.d_model)
        
        # Dropout layer used repeatedly
        self.dropout = Dropout(0.1)
        
        # Encoder Transformer blocks
        self.encoder_blocks = [
            TransformerBlock(self.num_heads, self.key_dim, self.d_ff, dropout_rate=0.1)
            for _ in range(self.n_layers)
        ]
        
        # Decoder Transformer blocks
        self.decoder_blocks = []
        for _ in range(self.n_layers):
            # Each decoder block has self-attention, cross-attention, and feedforward
            self_attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=0.1)
            cross_attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=0.1)
            ffn = tf.keras.Sequential([
                Dense(self.d_ff, activation='relu'),
                Dense(self.d_model)
            ])
            ln1 = LayerNormalization(epsilon=1e-6)
            ln2 = LayerNormalization(epsilon=1e-6)
            ln3 = LayerNormalization(epsilon=1e-6)
            dropout_layer = Dropout(0.1)
            self.decoder_blocks.append({
                "self_attn": self_attn,
                "cross_attn": cross_attn,
                "ffn": ffn,
                "ln1": ln1,
                "ln2": ln2,
                "ln3": ln3,
                "dropout": dropout_layer
            })
        
        # Final dense softmax output layer of decoder
        self.final_dense = Dense(self.decoder_vocab_len, activation='softmax', name='decoder_output')

    def call(self, inputs, training=None):
        # inputs is a tuple/list of (encoder_input, decoder_input)
        encoder_input, decoder_input = inputs
        
        # Encoder embedding + dropout
        enc_embedded = self.encoder_embedding(encoder_input)
        enc_embedded = self.dropout(enc_embedded, training=training)
        
        enc_output = enc_embedded
        for block in self.encoder_blocks:
            enc_output = block(enc_output, training=training)
        
        # Decoder embedding + dropout
        dec_embedded = self.decoder_embedding(decoder_input)
        dec_embedded = self.dropout(dec_embedded, training=training)
        
        dec_output = dec_embedded
        
        for block in self.decoder_blocks:
            # Self-attention
            self_attn_out = block["self_attn"](dec_output, dec_output, dec_output)
            self_attn_out = block["dropout"](self_attn_out, training=training)
            out1 = block["ln1"](dec_output + self_attn_out)
            
            # Cross-attention over encoder output
            cross_attn_out = block["cross_attn"](out1, enc_output, enc_output)
            cross_attn_out = block["dropout"](cross_attn_out, training=training)
            out2 = block["ln2"](out1 + cross_attn_out)
            
            # Feedforward
            ffn_out = block["ffn"](out2)
            ffn_out = block["dropout"](ffn_out, training=training)
            dec_output = block["ln3"](out2 + ffn_out)
        
        # Final output with softmax
        output = self.final_dense(dec_output)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a tuple of inputs matching (encoder_input, decoder_input)
    # According to example in issue:
    # encoder_input shape: (batch=2, seq_len=1)
    # decoder_input shape: (batch=2, seq_len=4)
    # Inputs are integer token ids ranging from 0 to vocab_size-1 (10000)
    batch_size = 2
    encoder_seq_len = 1
    decoder_seq_len = 4
    encoder_input = tf.random.uniform(
        shape=(batch_size, encoder_seq_len), minval=0, maxval=9999, dtype=tf.int32
    )
    decoder_input = tf.random.uniform(
        shape=(batch_size, decoder_seq_len), minval=0, maxval=9999, dtype=tf.int32
    )
    return (encoder_input, decoder_input)


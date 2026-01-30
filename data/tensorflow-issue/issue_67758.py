import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

    def call(self, inputs, training=None, *args, **kwargs):
        inputs_proj = self.dense_proj(inputs)
        print(f"inputs_proj shape: {inputs_proj.shape}")

        if len(inputs_proj.shape) == 4 and inputs_proj.shape[2] == 1:
            inputs_proj = tf.squeeze(inputs_proj, axis=2)
            print(f"inputs_proj after squeeze shape: {inputs_proj.shape}")

        attn_output = self.att(inputs_proj, inputs_proj)
        print(f"attn_output shape: {attn_output.shape}")
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs_proj + attn_output)
        ffn_output = self.ffn(out1)
        print(f"ffn_output shape: {ffn_output.shape}")
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DelayHead(tf.keras.layers.Layer):
    def __init__(self,
                 number_heads: int,
                 head_hidden_size: int,
                 initializer_range: float,
                 head_layer_norm_eps: float,
                 head_dropout_prob: float,
                 **kwargs):
        super().__init__(**kwargs)

        self.number_heads = number_heads
        self.denses = []
        self.layer_norms = []
        for _ in range(self.number_heads):
            self.denses.append(
                tf.keras.layers.Dense(
                    units=head_hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                    name='dense'
                )
            )
            self.layer_norms.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=head_layer_norm_eps,
                    name='layer_norm'
                )
            )
        self.dropout = tf.keras.layers.Dropout(
            rate=head_dropout_prob
        )
        self.labeling = tf.keras.layers.Dense(
            units=1,
            activation='linear',
            name='label'
        )

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        for i in range(self.number_heads):
            hidden_state = self.denses[i](hidden_state)
            hidden_state = self.layer_norms[i](hidden_state)
        hidden_state = self.dropout(hidden_state)
        return self.labeling(hidden_state)
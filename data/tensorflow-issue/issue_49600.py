# tf.random.uniform((B, 300), dtype=tf.float32)  # Assumed input shape: batch_size x 300 features

import tensorflow as tf

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
        for i in range(self.number_heads):
            self.denses.append(
                tf.keras.layers.Dense(
                    units=head_hidden_size,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
                    name=f'dense_{i}'
                )
            )
            self.layer_norms.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=head_layer_norm_eps,
                    name=f'layer_norm_{i}'
                )
            )
        self.dropout = tf.keras.layers.Dropout(rate=head_dropout_prob)
        self.labeling = tf.keras.layers.Dense(
            units=1,
            activation='linear',
            name='label'
        )

    def call(self, hidden_state: tf.Tensor, training=False) -> tf.Tensor:
        # Sequentially apply all heads' dense + layer_norm layers in order
        for i in range(self.number_heads):
            hidden_state = self.denses[i](hidden_state)
            hidden_state = self.layer_norms[i](hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        return self.labeling(hidden_state)


class MyModel(tf.keras.Model):
    def __init__(self,
                 number_heads=2,
                 head_hidden_size=128,
                 initializer_range=0.02,
                 head_layer_norm_eps=1e-7,
                 head_dropout_prob=0.1):
        """
        Constructs the model encapsulating the DelayHead.
        Args:
          number_heads: Number of heads (stacked dense + layer_norm layers).
          head_hidden_size: Hidden size for dense layers.
          initializer_range: Stddev for truncated normal initializer.
          head_layer_norm_eps: Epsilon for LayerNormalization.
          head_dropout_prob: Dropout rate.
        """
        super().__init__()
        self.delay_head = DelayHead(
            number_heads=number_heads,
            head_hidden_size=head_hidden_size,
            initializer_range=initializer_range,
            head_layer_norm_eps=head_layer_norm_eps,
            head_dropout_prob=head_dropout_prob,
            name='delay_head'
        )

    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        Args:
          inputs: Tensor of shape (batch_size, 300) assumed feature vector.
          training: Boolean, whether in training mode (affects dropout).
        Returns:
          Output tensor of shape (batch_size, 1).
        """
        # Just delegate to delay_head
        return self.delay_head(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel with default parameters matching the issue's description
    return MyModel(
        number_heads=2,
        head_hidden_size=128,
        initializer_range=0.02,
        head_layer_norm_eps=1e-7,  # Recommended epsilon that avoids the reported error
        head_dropout_prob=0.1
    )


def GetInput():
    """
    Returns a random input tensor matching the expected input shape of MyModel.

    Based on the issue:
    - Input vector size ~300 dimensions
      * First 256 dims from pooled CLS embedding (values roughly in [-1,1])
      * Next ~10 dims unscaled but positive (simulate by uniform [0,5])
      * Last dims embeddings from embedding layers (simulate normalized random)

    We'll concatenate these parts to produce a realistic synthetic input.
    """
    batch_size = 1  # Minimum batch size matching issue logs
    cls_features = tf.random.uniform((batch_size, 256), minval=-1.0, maxval=1.0, dtype=tf.float32)
    positive_features = tf.random.uniform((batch_size, 10), minval=0.0, maxval=5.0, dtype=tf.float32)
    embedding_features = tf.random.normal((batch_size, 34), mean=0.0, stddev=0.02, dtype=tf.float32)  # 300 - 256 -10 = 34 dims

    # Concatenate to form final input
    input_tensor = tf.concat([cls_features, positive_features, embedding_features], axis=-1)
    return input_tensor


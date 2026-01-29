# tf.random.uniform((B, size, size), dtype=tf.float32), tf.random.uniform((B, size), dtype=tf.int32)

import tensorflow as tf

class EmbeddingPostprocessor(tf.keras.layers.Layer):
    def __init__(self,
                 use_type_embeddings=True,
                 use_position_embeddings=True,
                 token_type_vocab_size=16,
                 type_embedding_width=128,
                 position_embedding_width=128,
                 max_position_embeddings=512,
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_type_embeddings = use_type_embeddings
        self.use_position_embeddings = use_position_embeddings
        self.token_type_vocab_size = token_type_vocab_size
        self.dtype = dtype

        # Create trainable embeddings
        if self.use_type_embeddings:
            self.type_embeddings = self.add_weight(
                name="type_embeddings",
                shape=[self.token_type_vocab_size, type_embedding_width],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                dtype=self.dtype,
                trainable=True,
            )

        if self.use_position_embeddings:
            self.position_embeddings = self.add_weight(
                name="position_embeddings",
                shape=[max_position_embeddings, position_embedding_width],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                dtype=self.dtype,
                trainable=True,
            )

        # Layer normalization and dropout as per BERT style
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, dtype=self.dtype
        )
        self.output_dropout = tf.keras.layers.Dropout(rate=0.1)

    def call(self, inputs):
        """
        inputs: a tuple or list: (word_embeddings, token_type_ids)
        word_embeddings shape: [batch_size, seq_length, width]
        token_type_ids shape: [batch_size, seq_length]
        """

        word_embeddings, token_type_ids = tf.nest.flatten(inputs)

        input_shape = tf.shape(word_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        output = word_embeddings

        if self.use_type_embeddings:
            # The "fix" from issue: use tf.gather instead of tf.matmul with one-hot
            # flatten for gather, then reshape back
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])  # [batch_size * seq_length]
            token_type_embeddings = tf.gather(self.type_embeddings, flat_token_type_ids)
            token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            output += token_type_embeddings

        if self.use_position_embeddings:
            # Slice only needed positions and expand dims for batch
            # Use seq_length to slice position embeddings
            position_embeddings = self.position_embeddings[:seq_length, :]
            position_embeddings = tf.expand_dims(position_embeddings, axis=0)
            output += position_embeddings

        output = self.output_layer_norm(output)
        output = self.output_dropout(output)

        return output


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on input examples, input word_embeddings has shape (B, size, size),
        # token_type_ids has shape (B, size), so set embedding width to size for compatibility.
        # Use EmbeddingPostprocessor with use_position_embeddings=True.
        self.postprocessor = EmbeddingPostprocessor(
            use_type_embeddings=True,
            use_position_embeddings=True,
            token_type_vocab_size=2,
            type_embedding_width=None,  # Will be inferred dynamically in build()
            position_embedding_width=None,
            dtype=tf.float32,
        )

    def build(self, input_shape):
        # Infer embedding widths from input shapes: input_shape = [(B, H, W), (B, H)]
        # We assume width for embeddings = input_shape[0][-1]
        word_embedding_shape = input_shape[0]
        width = word_embedding_shape[-1]

        # If type_embeddings or position_embeddings have None shape,
        # recreate them with correct width. This logic is needed as
        # EmbeddingPostprocessor __init__ doesn't know width upfront.
        # To keep it simple, reinit weights with tf.Variable here.
        # But since we already added weights in __init__, use workaround:
        def recreate_weights(layer, name, shape):
            # Remove old weights and add new weights of correct shape.
            # This is a logical placeholder - in real usage these should be created properly in build().
            del layer._trainable_weights[:]  # Clear weights
            weight = layer.add_weight(
                name=name,
                shape=shape,
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                dtype=layer.dtype,
                trainable=True,
            )
            return weight

        # Recreate or set type_embeddings with correct shape
        if self.postprocessor.use_type_embeddings:
            if self.postprocessor.type_embeddings.shape[-1] != width:
                self.postprocessor.type_embeddings = recreate_weights(
                    self.postprocessor, "type_embeddings", [self.postprocessor.token_type_vocab_size, width]
                )
        # Recreate or set position_embeddings with correct shape
        if self.postprocessor.use_position_embeddings:
            if self.postprocessor.position_embeddings.shape[-1] != width:
                self.postprocessor.position_embeddings = recreate_weights(
                    self.postprocessor, "position_embeddings", [512, width]
                )

        super().build(input_shape)

    def call(self, inputs):
        # inputs: tuple of (word_embeddings, token_type_ids)
        return self.postprocessor(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Produce input tensors matching expected input shapes from the issue:
    # From example:
    # input1 = shape (batch=1, size=100, size=100), dtype float32
    # input2 = shape (batch=1, size=100), dtype int32, values in [0,1]
    size = 100
    batch = 1

    word_embeddings = tf.random.uniform(
        (batch, size, size), minval=0, maxval=1, dtype=tf.float32
    )
    token_type_ids = tf.random.uniform(
        (batch, size), minval=0, maxval=2, dtype=tf.int32
    )
    return (word_embeddings, token_type_ids)


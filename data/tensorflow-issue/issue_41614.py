# tf.random.uniform((B, L), dtype=tf.int32) ← Input shape inferred as batch of input token IDs sequences (batch_size, sequence_length)

import tensorflow as tf

def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Configuration parameters inferred from the snippet
        self.vocab_size = 30522
        self.embedding_size = 768
        self.initializer_range = 0.02
        self.max_position_embeddings = 384
        self.type_vocab_size = 2

        # Embedding layers and variables
        self.position_embeddings = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings",
        )
        # LayerNorm and Dropout
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(0.1)
        
    def build(self, input_shape):
        """Create and initialize the shared word embedding variable"""
        with tf.name_scope("word_embeddings"):
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
                dtype=tf.float32,  # embeddings usually float32 variables
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs, training=True):
        """
        Inputs is expected to be a list or tuple:
        [input_ids, position_ids, token_type_ids, inputs_embeds]

        - input_ids: int tensor [batch, seq]
        - position_ids: int tensor [batch, seq] or None
        - token_type_ids: int tensor [batch, seq] or None
        - inputs_embeds: float tensor [batch, seq, embedding_size] or None

        Returns:
            embedded and normalized tensor of shape [batch, seq, embedding_size]
            with dtype float16 if mixed_precision policy is active.
        """
        return self._embedding(inputs, training=training)

    def _embedding(self, inputs, training=False):
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]
        
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        if inputs_embeds is None:
            # Gather embeddings for each input id (float32), then mixed_precision casting will happen later as needed
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Explicitly cast embeddings to mixed precision dtype (usually float16) to avoid add type mismatch error.
        # This follows the resolution shared in the issue.
        # inputs_embeds is float32 (variable), cast to float16 if mixed precision active
        dtype_policy = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
        if dtype_policy == 'float16':
            inputs_embeds = tf.cast(inputs_embeds, tf.float16)
            position_embeddings = tf.cast(position_embeddings, tf.float16)
            token_type_embeddings = tf.cast(token_type_embeddings, tf.float16)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a typical input matching the expected input for MyModel

    batch_size = 2
    seq_length = 5

    # input_ids: integer token ids, shape (batch_size, seq_length)
    input_ids = tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=30522,
        dtype=tf.int32,
    )
    # position_ids can be None to test default range creation in model
    position_ids = None

    # token_type_ids: int32, shape (batch_size, seq_length), values in 0..1 for two token types
    token_type_ids = tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=2,
        dtype=tf.int32,
    )

    # inputs_embeds: None so model gathers embeddings from input_ids
    inputs_embeds = None

    return [input_ids, position_ids, token_type_ids, inputs_embeds]


# tf.random.uniform((B, SeqLen), dtype=tf.int64) and (B, SeqLen) position ids as inputs

import tensorflow as tf
import numpy as np

def positional_encoding(shape, dtype=tf.float32):
    """
    Positional encoding initializer as used in the CustomEmbedding layer.
    Generates sinusoidal embeddings for position indices.
    
    Args:
        shape: tuple of (max_position_embeddings, hidden_size)
        dtype: output dtype
    
    Returns:
        Tensor of shape (max_position_embeddings, hidden_size) with positional encodings.
    """
    n_pos, dim = shape
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])

    # apply sin to even indices in the array; 2i
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    return tf.cast(position_enc, dtype=dtype)


class CustomEmbedding(tf.keras.layers.Layer):
    """
    Custom embedding layer combining word embeddings and positional embeddings.
    Positional embeddings are initialized with a fixed positional_encoding initializer.
    Word embeddings are trainable.
    """
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # +1 to max_position_embeddings as per original code
        self.max_position_embeddings = max_position_embeddings + 1

    def build(self, input_shape):
        with tf.name_scope("position_embeddings"):
            # Important: Provide a 'name' to the weight to avoid checkpoint saving issue
            self.position_embeddings = self.add_weight(
                name="position_embeddings",
                shape=(self.max_position_embeddings, self.hidden_size),
                initializer=positional_encoding,
                trainable=False,
                dtype=self.dtype)

        with tf.name_scope("word_embeddings"):
            self.word_embeddings = self.add_weight(
                name="token_weight",
                shape=(self.vocab_size, self.hidden_size),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
                dtype=self.dtype
            )
        super(CustomEmbedding, self).build(input_shape)

    def call(self, inputs):
        input_ids, position_ids = inputs
        inputs_embeds = tf.nn.embedding_lookup(self.word_embeddings, input_ids)
        position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class TestModel(tf.keras.Model):
    """
    A simple classifier model that uses CustomEmbedding and a Dense layer.
    Takes as input a tuple of (input_ids, position_ids), embeds them and predicts classes.
    """
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, num_class, **kwargs):
        super(TestModel, self).__init__(**kwargs)
        self.emb = CustomEmbedding(vocab_size, hidden_size, max_position_embeddings)
        self.dense = tf.keras.layers.Dense(num_class, name="class_prj")

    def call(self, inputs):
        word_emb = self.emb(inputs)
        # Mean pooling over sequence length dimension (axis=1)
        sent_emb = tf.reduce_mean(word_emb, axis=1)
        logit = self.dense(sent_emb)
        return logit


class MyModel(tf.keras.Model):
    """
    Wrapper model that encapsulates the TestModel with fixed hyperparameters.
    This acts as the main entrypoint model as requested.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # Using same hyperparameters as the provided example
        VOCAB_SIZE = 100
        HIDDEN_SIZE = 5
        MAX_POSITION_EMBEDDING = 30
        NUM_CLASS = 3
        self.model = TestModel(VOCAB_SIZE, HIDDEN_SIZE, MAX_POSITION_EMBEDDING, NUM_CLASS)

    def call(self, inputs):
        return self.model(inputs)


def my_model_function():
    """
    Factory function to return an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Generate dummy inputs corresponding to the input signature expected by MyModel:
    - input_ids: (batch_size, seq_len), int64
    - position_ids: (batch_size, seq_len), int64 (positions from 0 to seq_len-1)
    
    We choose:
      batch_size = 3
      seq_len = 4
    
    Returns:
      Tuple of two tf.Tensor objects (input_ids, position_ids)
    """
    batch_size = 3
    seq_len = 4
    vocab_size = 100  # must <= MyModel VOCAB_SIZE

    # Random input_ids in vocabulary range [0, vocab_size)
    input_ids = tf.random.uniform(
        shape=(batch_size, seq_len),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int64)

    # position ids from 0 to seq_len-1 repeated for each batch
    position_ids = tf.tile(
        tf.expand_dims(tf.range(seq_len, dtype=tf.int64), 0),
        multiples=[batch_size, 1])

    return (input_ids, position_ids)


# tf.RaggedTensor with shape (batch_size, 400, 3, None), dtype=tf.int32

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, LSTM, MaxPool1D, Dense

class EnhancedEmbedding(tf.keras.layers.Embedding):
    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer='uniform',
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        **kwargs
    ):
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer,
            embeddings_regularizer,
            activity_regularizer,
            embeddings_constraint,
            mask_zero,
            input_length,
            **kwargs
        )
        self.five = tf.constant(0.5, dtype=tf.float32)
        self.zero = tf.constant(0, dtype=tf.int32)

    def embedding(self, inputs):
        # Use parent's call method to get embeddings for input token ids
        return super().call(inputs)

    def map_2(self, tokens):
        # tokens is expected shape: (3, None) ragged slice: tokens[0], tokens[2] are token indices
        # Wehed the shape condition to handle case based on shape of tokens[0]
        # If tokens[0] shape is zero (empty), return cur_word squeezed,
        # else return weighted avg of mean(identifier)*0.5 + cur_word*0.5
        identifier = self.embedding(tokens[0])  # embedding for tokens[0]
        cur_word = self.embedding(tokens[2])    # embedding for tokens[2]

        # tf.cond expects predicate scalar bool, but tf.shape returns shape vector,
        # so we compare tf.shape(tokens[0])[0] with zero.
        len_id = tf.shape(tokens[0])[0]

        def true_fn():
            # if length identifier is zero, squeeze cur_word
            return tf.squeeze(cur_word)

        def false_fn():
            # else: weighted average
            mean_id = tf.reduce_mean(identifier)
            return tf.squeeze(mean_id * self.five + cur_word * self.five)

        return tf.cond(tf.equal(len_id, self.zero), true_fn, false_fn)

    def map_1(self, inputs):
        # inputs shape: (400, 3, None) - apply map_2 to each tokens input (dim=3, None)
        # map_2 expects tokens of shape (3, None),
        # but map_fn passes 1D slices, so axis handling is critical.
        # We map over inputs, which has shape (400, 3, None),
        # so elems=inputs maps over dim-0 (400 tokens).
        return tf.map_fn(fn=lambda x: self.map_2(x), elems=inputs, dtype=tf.float32)

    def call(self, inputs):
        # inputs is RaggedTensor (batch_size, 400, 3, None)
        # Apply map_1 over batch dimension (elems=inputs)
        # map_1 returns (400, embedding_dim), so final shape (batch_size, 400, embedding_dim)
        final_embeddings = tf.map_fn(fn=lambda x: self.map_1(x), elems=inputs, dtype=tf.float32)
        return final_embeddings


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumed constants for dimensions (can be changed as needed)
        self.embedding_dim = 32
        self.hidden_dim = 64
        self.vocab_size = 10000
        self.label_size = 10
        self.seq_len = 400

        # Placeholder pretrained weights initializer - uniform random for example
        pretrained_weight = tf.random.uniform(
            shape=(self.vocab_size, self.embedding_dim), dtype=tf.float32
        )

        self.embedding = EnhancedEmbedding(
            self.vocab_size,
            self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(pretrained_weight),
        )
        self.encoder = Bidirectional(
            LSTM(self.hidden_dim, return_sequences=True, input_shape=(self.seq_len, self.embedding_dim))
        )
        # MaxPool1D over last dimension, pool size = hidden_dim*2 because of bidirectional
        self.pool = MaxPool1D(pool_size=self.hidden_dim * 2)
        self.decoder = Dense(self.label_size)

    def call(self, inputs):
        # inputs : RaggedTensor of shape (batch_size, 400, 3, None), dtype int32
        embeddings = self.embedding(inputs)  # (batch_size, 400, embedding_dim)

        # Explicitly reshape or ensure embeddings shape to (batch_size, 400, embedding_dim)
        # According to original code, reshape to [-1, 400, 32]
        embeddings = tf.reshape(embeddings, [-1, self.seq_len, self.embedding_dim])

        lstm_out = self.encoder(embeddings)  # (batch_size, 400, hidden_dim*2)

        # transpose to (batch_size, hidden_dim*2, 400) for MaxPool1D
        lstm_out = tf.transpose(lstm_out, perm=[0, 2, 1])

        pool_out = self.pool(lstm_out)  # (batch_size, hidden_dim*2, 1)

        # Squeeze pool dimension
        out = tf.squeeze(pool_out, axis=[2])

        out = self.decoder(out)  # (batch_size, label_size)

        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a RaggedTensor input with shape (batch_size, 400, 3, None), dtype int32
    # The last dimension None can be random length per slice,
    # For example, demo with batch_size=2, each token triple with variable length 1-3.

    batch_size = 2
    seq_len = 400
    triples_per_token = 3

    # Construct ragged tensor input as nested lists of ints.
    # For demonstration, each innermost element is a list of integers (token indices).
    # We'll create ragged lists with random lengths 1-3.
    import numpy as np

    # Helper to generate random list of ints with variable length from 1 to 3
    def random_token_list():
        length = np.random.randint(1, 4)
        return np.random.randint(0, 9999, size=(length,)).tolist()

    inputs_data = []
    for _ in range(batch_size):
        seq_data = []
        for _ in range(seq_len):
            triple = [random_token_list() for _ in range(triples_per_token)]
            seq_data.append(triple)
        inputs_data.append(seq_data)

    # Convert nested python lists to RaggedTensor
    # RaggedTensor will have ragged dimension only at last level (None)
    return tf.ragged.constant(inputs_data, dtype=tf.int32)


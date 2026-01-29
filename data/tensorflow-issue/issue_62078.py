# tf.ragged.constant([[...]], dtype=tf.int32) ← RaggedTensor of variable-length sequences as input IDs with shape (batch_size, None)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding lookup weights initialization - assuming fixed vocab size 16 and embedding dim 4 as in example
        self.embedding_dim = 4
        self.vocab_size = 16
        # Create embedding weights as a trainable variable to simulate embedding matrix
        self.weights = tf.Variable(
            initial_value=tf.random.uniform(shape=(self.vocab_size, self.embedding_dim), minval=0, maxval=1),
            trainable=True,
            name="embedding_weights"
        )

    def call(self, lookup_ids):
        """
        Perform embedding lookup using tf.nn.safe_embedding_lookup_sparse on RaggedTensor lookup_ids.

        Args:
            lookup_ids: tf.RaggedTensor of shape (batch_size, None), dtype=tf.int32 representing IDs to lookup.

        Returns:
            A RaggedTensor of embeddings, shape (batch_size, None, embedding_dim).
        """
        # tf.nn.safe_embedding_lookup_sparse expects SparseTensor for ids argument.
        # RaggedTensor can be converted to SparseTensor without materializing dense shape (RaggedTensor.to_sparse()).
        # According to the issue discussion, direct RaggedTensor may work in newer TF versions,
        # but underlying it may still transform to SparseTensor.

        # Convert RaggedTensor to SparseTensor to use safe_embedding_lookup_sparse
        sparse_ids = lookup_ids.to_sparse()

        # Perform safe embedding lookup — this returns a dense Tensor with shape (batch_size, embedding_dim).
        # However, safe_embedding_lookup_sparse expects weights and SparseTensor ids and returns embeddings shape (batch_size, embedding_dim).
        # But since our SparseTensor has ragged rows (different lengths), tf.nn.safe_embedding_lookup_sparse aggregates embeddings per row.
        # This matches in typical use when SparseTensor is a "bag" of ids per batch element.
        # The output is aggregated embeddings per batch element.

        # To preserve per-token embeddings, we need to do a gather instead.
        # But the issue is about usage of safe_embedding_lookup_sparse on batched RaggedTensor ids.
        # So here we showcase the typical usage: aggregated embeddings per batch row.

        embeddings = tf.nn.safe_embedding_lookup_sparse(self.weights, sparse_ids, default_id=-1)
        # embeddings shape: (batch_size, embedding_dim)

        return embeddings

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Produce example input: a RaggedTensor representing a batch of sequences of IDs.
    # This matches the example from the issue discussion.
    # Batch size = 3, variable-length sequences

    input_data = tf.ragged.constant([[1, 2, 3], [4, 5], [6]], dtype=tf.int32)
    return input_data


# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape is (batch_size, variable_length_sequence)

import tensorflow as tf

class EmbeddingMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EmbeddingMean, self).__init__(**kwargs)
        # Note: supports_ragged_inputs attribute was removed in TF nightly versions as per issue comment
        # But to keep compatibility we implement call accordingly - no explicit attribute needed

    def call(self, inputs, **kwargs):
        # inputs: could be ragged or dense tensor of shape (batch_size, seq_len, embedding_dim)
        # reduce mean over axis=1 (sequence length axis)
        return tf.reduce_mean(inputs, axis=1)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding layer with vocab size 10 and embedding dim 3
        self.embedding = tf.keras.layers.Embedding(10, 3)
        # Custom layer defined above
        self.embedding_mean = EmbeddingMean()
        # Output Dense layer producing output scalar per batch element
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # inputs is a RaggedTensor or dense int32 tensor of shape (batch_size, None)
        x = self.embedding(inputs)  # (batch_size, seq_len, 3)
        x = self.embedding_mean(x)  # (batch_size, 3)
        output = self.output_layer(x)  # (batch_size, 1)
        return output

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return ragged int32 tensor input with shape (batch_size, variable length sequence)
    # Creating a ragged tensor with shape (3, None)
    # Example: batch of 3 sequences with lengths 4, 2, and 3 respectively
    rt = tf.ragged.constant([
        [1, 2, 3, 4],
        [3, 2],
        [1, 0, 4]
    ], dtype=tf.int32)
    return rt


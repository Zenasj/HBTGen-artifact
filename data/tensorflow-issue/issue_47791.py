# tf.random.uniform((B=10, H=152), dtype=tf.int32) ← input is a 2D integer tensor of shape (batch_size=10, sequence_length=152)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=8, embed_dim=64, maxlen=152):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        # Embedding layer with input_dim=vocab_size (no mask token)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            input_length=self.maxlen,
            name="embedding_layer"
        )

    def call(self, inputs, training=False):
        """
        Expects inputs as integer tensor of shape (batch_size, maxlen)
        Embedding lookup will fail with InvalidArgumentError if indices are out of range [0, vocab_size).
        """
        # Directly call embedding layer on inputs
        return self.embedding(inputs)

def my_model_function():
    # Returns an instance of MyModel initialized with vocab_size=8, embed_dim=64, maxlen=152
    return MyModel()

def GetInput():
    # Generate input tensor of shape (10, 152) with valid indices in [1, vocab_size]
    # Here vocab_size=8, indices range: [0..7] valid. Using lower bound 1 to mimic example but 0 is valid too.
    # This matches the example from issue (10 batch, seq_len=152)
    vocab_size = 8
    batch_size = 10
    maxlen = 152
    # Use tf.random.uniform to generate int32 tensor with values in valid vocab range [1, vocab_size - 1] inclusive
    # Since embedding layer expects indices in [0, vocab_size), avoid out of range values here
    return tf.random.uniform(
        shape=(batch_size, maxlen),
        minval=1,
        maxval=vocab_size,
        dtype=tf.int32
    )


# tf.random.uniform((B, seq_len), dtype=tf.int32) ← Input shape is (batch_size, sequence_length)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=10, embedding_dim=5, seq_len=20):
        super().__init__()
        # Embedding layer to map input token indices to vectors
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Time-distributed Dense layer with softmax activation to predict token probabilities per time step
        # Note: The example uses a Dense layer expecting input shape (B, seq_len, emb_dim)
        # and outputs (B, seq_len, vocab_size).
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.seq_len = seq_len

    def call(self, inputs):
        """
        inputs: an int32 tensor of shape (batch_size, seq_len) with token indices
        returns: tensor of shape (batch_size, seq_len, vocab_size) with softmax probabilities
        """
        x = self.embedding(inputs)  # shape: (B, seq_len, embedding_dim)
        output = self.dense(x)      # shape: (B, seq_len, vocab_size)
        return output

def my_model_function():
    # Returns an instance of MyModel with default parameters matching example usage
    return MyModel()

def GetInput():
    # Returns a random integer tensor of shape (batch_size=32, seq_len=20) with token ids in [0, vocab_size)
    batch_size = 32
    seq_len = 20
    vocab_size = 10
    # Using uniform random ints to simulate token sequences for input
    return tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)


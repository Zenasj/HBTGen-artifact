# tf.random.uniform((B, 10), dtype=tf.int32) ← Infer input shape as sequences of length 10 (word indices)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=5000, embedding_dim=100, embedding_matrix=None):
        super().__init__()
        # Use provided embedding matrix or random initialization if None
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=10,
                trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=10)

        # CuDNNLSTM was used in TF 2.0 alpha for GPU-accelerated LSTM - 
        # now replaced by standard LSTM with default configurations
        self.lstm = tf.keras.layers.LSTM(50)  

        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)  # (batch, 10, embedding_dim)
        x = self.lstm(x)            # (batch, 50)
        x = self.dense1(x)          # (batch, 50)
        x = self.flatten(x)         # (batch, 50)
        x = self.dropout(x, training=training)
        x = self.out_dense(x)       # (batch, 1)
        return x

def my_model_function():
    # For demonstration, initialize with default vocab size and random embeddings.
    # In practice, user should provide real vocab_size and embedding_matrix.
    return MyModel()

def GetInput():
    # Random integer tensor batch simulating input sequences of word indices
    batch_size = 32
    vocab_size = 5000  # Should match model vocab size
    sequence_length = 10
    # Inputs must be int32 for Embedding layer
    return tf.random.uniform((batch_size, sequence_length), minval=0, maxval=vocab_size, dtype=tf.int32)


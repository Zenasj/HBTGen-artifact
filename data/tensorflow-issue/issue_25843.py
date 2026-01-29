# tf.random.uniform((B, T), dtype=tf.int32) ← Input shape inferred from dataset1 as sequences of token ids (shape: batch_size x sequence_length)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=10, embedding_dim=80, lstm_units=64):
        super().__init__()
        # Embedding layer: converts integer sequences to dense vectors
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # LSTM layer: unified LSTM, will use GPU-accelerated CuDNN-based kernel if environment allows
        self.lstm = tf.keras.layers.LSTM(lstm_units,
                                         activation='tanh',           # required for GPU optimized usage
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,
                                         use_bias=True)
        # Output dense sigmoid layer for binary classification
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default parameters suitable for demo
    return MyModel()

def GetInput():
    # Generate a random batch of integer token sequences, matching the input shape and type expected by MyModel
    # Here we pick batch size = 2 and sequence length = 3, based on dataset from the issue example
    batch_size = 2
    sequence_length = 3
    vocab_size = 10
    # dtype=int32 for token indices
    return tf.random.uniform((batch_size, sequence_length), minval=0, maxval=vocab_size, dtype=tf.int32)


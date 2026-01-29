# tf.random.uniform((B, seq_len), dtype=tf.float32) ← Inferred input shape from example "Predict Shakespeare with Cloud TPUs and Keras" (batch_size B, sequence length)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, seq_len=100, batch_size=128, stateful=False, units=256, vocab_size=65):
        super().__init__()
        # A simple LSTM model inspired by "Predict Shakespeare with Cloud TPUs and Keras"
        # Input: (batch_size, seq_len) - sequence of integer character indices
        # Output: logits for next character prediction
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.stateful = stateful

        self.embedding = tf.keras.layers.Embedding(vocab_size, units)
        self.lstm = tf.keras.layers.LSTM(units, stateful=stateful, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # inputs: int32 tensor of shape (batch_size, seq_len)
        x = self.embedding(inputs)  # (batch_size, seq_len, units)
        x = self.lstm(x)            # (batch_size, seq_len, units)
        logits = self.dense(x)      # (batch_size, seq_len, vocab_size)
        return logits

def my_model_function():
    # Return an instance of MyModel with default parameters matching the TPU example
    return MyModel()

def GetInput():
    # Return a random input tensor matching MyModel's expected input:
    # integer token sequences, shape (batch_size, seq_len), values in [0, vocab_size)
    batch_size = 128  # from example
    seq_len = 100
    vocab_size = 65   # typical Shakespeare character vocab size
    
    return tf.random.uniform(
        shape=(batch_size, seq_len),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )


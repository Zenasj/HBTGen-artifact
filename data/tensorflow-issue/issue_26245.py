# tf.random.uniform((B, maxlen), dtype=tf.int32) ← Input is padded integer sequences of shape (batch_size, maxlen)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=20000, maxlen=80, embedding_dim=100, lstm_units=64):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen)
        # Note: recurrent_dropout causes issues with multi-GPU and CuDNN; we omit it
        self.lstm = tf.keras.layers.LSTM(
            lstm_units, dropout=0.2, recurrent_dropout=0.0)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x, training=training)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel with default parameters as in the original example
    return MyModel()

def GetInput():
    # Return a batch of random integer sequences matching the input shape expected by MyModel:
    # shape = (batch_size, maxlen), dtype = int32 for embedding inputs
    batch_size = 32    # Typical smaller batch; original batch 2048 too large for example here
    maxlen = 80        # Sequence length as per original code
    vocab_size = 20000 # Vocabulary size as in the original example
    
    # Generate random integer sequences in [1, vocab_size) to simulate word indices
    return tf.random.uniform(
        shape=(batch_size, maxlen), minval=1, maxval=vocab_size, dtype=tf.int32)


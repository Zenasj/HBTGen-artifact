# tf.random.uniform((B, 5), dtype=tf.int32) ← input is batch of sequences of length 5 with integer word indices

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, words_count=11111, embedding_dim=32, lstm_units=32):
        super().__init__()
        # Embedding layer with mask_zero=True to enable masking (cause of cudnn issue if batch too large)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=words_count,
            output_dim=embedding_dim,
            input_length=5,
            mask_zero=True,
            name="embedding"
        )
        # CuDNN-accelerated LSTM layer - This uses the fused cuDNN kernel under the hood if GPU available.
        self.lstm = tf.keras.layers.LSTM(
            lstm_units,
            name="lstm"
        )
    
    def call(self, inputs, training=False):
        # inputs: tensor of shape (batch_size, 5) - integer word indices
        x = self.embedding(inputs)  # shape (batch_size, 5, embedding_dim)
        x = self.lstm(x)            # shape (batch_size, lstm_units)
        return x

def my_model_function():
    # Create and return a new instance of MyModel
    # Use default vocab size 11111, embedding dim 32, LSTM units 32 as per issue
    return MyModel()

def GetInput():
    # Return a random integer tensor shaped (batch_size, 5) suitable for input_words Input layer.
    # Batch size chosen as 128 (common batch size from the issue),
    # vocab size is 11111, so values in range [1, 11110] because mask_zero=True reserves 0 for masking.
    batch_size = 128
    sequence_length = 5
    vocab_size = 11111

    # Random integers for word indices excluding 0 (mask token)
    input_tensor = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=1,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return input_tensor


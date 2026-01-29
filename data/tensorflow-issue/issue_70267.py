# tf.random.uniform((B, H), dtype=tf.int32)  ← Input is sequence of token IDs, shape = (batch_size, sequence_length)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, max_sequence_len, total_words):
        super().__init__()
        # Embedding layer converts word indices into dense vectors
        self.embedding = tf.keras.layers.Embedding(input_dim=total_words, output_dim=10,
                                                   input_length=max_sequence_len - 1)
        # LSTM layer with 150 units to capture sequence relationships
        self.lstm = tf.keras.layers.LSTM(150)
        # Dense output layer with softmax activation for multi-class classification (next word prediction)
        self.dense = tf.keras.layers.Dense(total_words, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x)
        output = self.dense(x)
        return output

def my_model_function(max_sequence_len=None, total_words=None):
    # Require max_sequence_len and total_words to initialize model appropriately
    # The values should come from data preprocessing steps
    return MyModel(max_sequence_len, total_words)

def GetInput():
    # Assumptions:
    # max_sequence_len = 5 (arbitrary small length based on typical example)
    # total_words = 30 (vocabulary size)
    # We'll generate random integer token indices in [1, total_words)
    max_sequence_len = 5
    total_words = 30
    # Return shape: (batch_size=2, sequence_length=max_sequence_len-1)
    batch_size = 2
    sequence_length = max_sequence_len - 1  # input excludes last token for label

    random_input = tf.random.uniform(shape=(batch_size, sequence_length),
                                     minval=1,
                                     maxval=total_words,
                                     dtype=tf.int32)
    return random_input


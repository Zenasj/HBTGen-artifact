# tf.random.uniform((B, max_len), dtype=tf.int32) ← Input shape is (batch_size, sequence_length)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=50, embedding_dim=32, max_len=9):
        super().__init__()
        # Based on the code in the issue:
        # vocab_size=50, embedding_dim=32 (changed from toy example),
        # max_len inferred from max length of padded sequences (original max_len was max_len + 2)
        # We keep max_len=9 from tensor example: max_len = max(len(x) for x in seqs)+2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(32)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel using the same vocab_size and max_len as in the example
    vocab_size = 50
    # max_len inferred from example docs padding, for robustness we set max_len=9 as in example
    max_len = 9
    embedding_dim = 32
    return MyModel(vocab_size=vocab_size, embedding_dim=embedding_dim, max_len=max_len)

def GetInput():
    # Returns a random int32 tensor shaped (batch_size, max_len) with vocab indices for the model input
    batch_size = 14  # same as training samples in the issue
    max_len = 9      # must match model input_length
    vocab_size = 50  # must match model vocab_size
    # Generate random integer sequences between 1 and vocab_size-1 (0 is usually reserved for padding)
    # dtype int32 as required by Embedding layer input
    return tf.random.uniform((batch_size, max_len), minval=1, maxval=vocab_size, dtype=tf.int32)


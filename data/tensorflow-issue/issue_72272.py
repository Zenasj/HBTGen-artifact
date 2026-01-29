# tf.random.uniform((B, 256), dtype=tf.int32)  ← Input shape inferred from model Input([256], dtype="int32")

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with input_dim=35000 and output_dim=10
        self.embedding = tf.keras.layers.Embedding(input_dim=35000, output_dim=10)
        # GRU layer with 10 units
        self.gru = tf.keras.layers.GRU(10)
        # Dense layer with 2 output units and softmax activation
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.gru(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Returns an instance of MyModel; weights uninitialized unless externally loaded
    return MyModel()

def GetInput():
    # Generate a random int32 tensor input with batch size 1 and sequence length 256
    # Matches Input([256], dtype="int32") of the original keras model
    # Vocabulary indices range from 0 to 34999 as per Embedding input_dim
    batch_size = 1
    sequence_length = 256
    vocab_size = 35000
    input_tensor = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return input_tensor


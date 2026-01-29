# tf.random.uniform((batch_size, sequence_length), dtype=tf.int32)  # Input shape inferred from IMDB dataset preprocessing: (batch=128, sequence=500)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: input_dim=10000 (vocab size), output_dim=10 (embedding size)
        self.embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=10)
        
        # SimpleRNN layer with 32 units as in original model
        self.simple_rnn = tf.keras.layers.SimpleRNN(32)
        
        # Dense output layer with sigmoid activation for binary classification
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
    
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        
        inputs: tf.Tensor of shape (batch_size, sequence_length), dtype int32 (word indices)
        returns: tf.Tensor of shape (batch_size, 1), probabilities between 0 and 1
        """
        x = self.embedding(inputs)          # Shape: (batch_size, sequence_length, 10)
        x = self.simple_rnn(x)             # Shape: (batch_size, 32)
        output = self.dense(x)             # Shape: (batch_size, 1)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random integer tensor representing padded word sequences
    # According to provided dataset setup:
    # batch size: 128 (typical batch size from example)
    # sequence length: 500 (maxlen parameter)
    # word indices range: [0, 9999] (num_words=10000)
    batch_size = 128
    sequence_length = 500
    vocab_size = 10000
    # Generate random indices in [0, vocab_size-1]
    return tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )


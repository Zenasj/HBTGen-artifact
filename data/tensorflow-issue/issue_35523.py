# tf.random.uniform((B, pad2), dtype=tf.int32) ← Input is a batch of padded sequences of token indices (integers)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, pad2, output_dim):
        super(MyModel, self).__init__()
        # Embedding layer initialized with pretrained embedding_matrix (300-dim)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            input_length=pad2,
            trainable=True
        )
        # First GRU layer with return_sequences=True
        self.gru1 = tf.keras.layers.GRU(128, return_sequences=True)
        # MaxPooling1D - reduce temporal dimension
        self.max_pool = tf.keras.layers.MaxPooling1D()
        # AveragePooling1D - reduce temporal dimension after max pooling
        self.avg_pool = tf.keras.layers.AveragePooling1D()
        # Second GRU layer without return_sequences (default is False)
        self.gru2 = tf.keras.layers.GRU(128)
        # Final Dense layer with sigmoid activation for multi-label classification
        self.dense = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)           # Shape: (batch, pad2, embedding_dim)
        x = self.gru1(x)                    # Shape: (batch, sequence_len, 128)
        x = self.max_pool(x)                # Shape: (batch, reduced_len, 128)
        x = self.avg_pool(x)                # Shape: (batch, further_reduced_len, 128)
        x = self.gru2(x)                   # Shape: (batch, 128)
        output = self.dense(x)             # Shape: (batch, output_dim)
        return output

def my_model_function():
    # Note: 
    # Since original code references "vocab_size", "dimensions", "embedding_matrix", "pad2", "outputs"
    # and their example initialization:
    #
    # vocab_size = len(tokenizer.word_index) + 1  # from dataset
    # dimensions = 300
    # embedding_matrix shape = (vocab_size, dimensions)
    # pad2 = maximum length of padded sequences
    # outputs = number of labels (classification output dimension)
    #
    # For this function, we set dummy sizes as placeholders.
    # In practical usage, these must be replaced with actual values/data.
    #
    # To keep the model instantiable without external data, initialize embedding_matrix with zeros.
    vocab_size = 10000    # guessed default vocab size
    dimensions = 300      # embedding dimension from example
    pad2 = 100            # maximum length of padded input sequences (max_len)
    output_dim = 10       # number of labels, assumed

    # Create dummy embedding matrix filled with zeros (as example)
    embedding_matrix = np.zeros((vocab_size, dimensions), dtype=np.float32)

    model = MyModel(vocab_size, dimensions, embedding_matrix, pad2, output_dim)

    # Build the model by calling it once on a dummy tensor
    dummy_input = tf.zeros((1, pad2), dtype=tf.int32)
    _ = model(dummy_input)

    return model

def GetInput():
    # Generate a random batch of padded sequences (batch size 4 for example)
    # Each element is an integer token index in [0, vocab_size-1]
    batch_size = 4
    pad2 = 100       # same max length as model expects
    vocab_size = 10000

    # Generate random integers in [1, vocab_size-1]; 0 can be reserved as padding token
    inputs = tf.random.uniform((batch_size, pad2), minval=1, maxval=vocab_size, dtype=tf.int32)
    return inputs


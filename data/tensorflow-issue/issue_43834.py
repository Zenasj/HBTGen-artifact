# tf.random.uniform((B,), dtype=tf.int32) ← Input shape is a 1D tensor of integers representing token IDs (as in the dataset and IntegerLookup vocabulary)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Vocabulary tokens from 1 to 10 inclusive
        vocabulary = tf.range(1, 11, dtype=tf.int64)  # IntegerLookup expects int64 for vocab
        
        # IntegerLookup layer to convert input tokens to indices
        self.lookup = tf.keras.layers.experimental.preprocessing.IntegerLookup(
            vocabulary=vocabulary.numpy(), mask_token=None, oov_token=None
        )
        
        # Embedding layer: vocab size + 2 for special tokens (mask and OOV)
        # Embedding output dim 8, input_length=1 as per original example
        self.embedding = tf.keras.layers.Embedding(input_dim=len(vocabulary) + 2, output_dim=8, input_length=1)
        
        # Output dense layer with sigmoid activation for binary classification
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.lookup(inputs)  # Map input tokens to indices
        x = self.embedding(x)    # Embed the indices
        x = tf.squeeze(x, axis=1)  # Remove the length=1 dimension for embedding
        x = self.dense(x)        # Binary prediction output
        return x

def my_model_function():
    # Return an instance of MyModel - no special weight loading needed
    return MyModel()

def GetInput():
    # Return a random integer tensor input matching expected input shape and dtype
    # Batch size 32 (arbitrary), single token per input (length=1)
    batch_size = 32
    # Input tokens are integers in range [1, 10]
    # Use tf.random.uniform to generate integer tokens between 1 and 10 inclusive
    return tf.random.uniform(shape=(batch_size,), minval=1, maxval=11, dtype=tf.int32)


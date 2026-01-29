# tf.random.uniform((GLOBAL_BATCH_SIZE,), dtype=tf.int32) ← inferred input shape is (batch_size * num_workers,), scalar integer tokens in [1,N_CAT)

import tensorflow as tf

N_CAT = 47  # Vocabulary size or number of categories (as per the issue)
EMBED_DIM = 5  # Embedding dimension used in the example

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer as per failing example
        self.embedding = tf.keras.layers.Embedding(input_dim=N_CAT, output_dim=EMBED_DIM)
        # Dense layer after embedding (note: in original code Dense without activation)
        self.dense = tf.keras.layers.Dense(N_CAT)
    
    def call(self, inputs):
        """
        inputs: shape (batch_size, 1), dtype int32 for categorical indices
        Returns logits over N_CAT classes, shape (batch_size, 1, N_CAT)
        """
        # inputs expected to be shape (batch_size, 1) - scalar token per batch element
        x = self.embedding(inputs)  # shape: (batch_size, 1, EMBED_DIM)
        x = self.dense(x)           # shape: (batch_size, 1, N_CAT)
        # Output shape is (batch_size, 1, N_CAT)
        return x

def my_model_function():
    # Return an instance of MyModel - no additional weights to load
    return MyModel()

def GetInput():
    # Create a random input tensor compatible with MyModel call:
    # Input shape is (batch_size, 1), dtype int32 integers in [1, N_CAT)
    # Batch size is not specified exactly, so choose a batch size of 32 as per example
    batch_size = 32
    input_tensor = tf.random.uniform(
        shape=(batch_size, 1), minval=1, maxval=N_CAT, dtype=tf.int32)
    return input_tensor


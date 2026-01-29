# tf.random.uniform((B, None), dtype=tf.int32) ← inferred input shape: batch dimension B and variable sequence length None (1D sequence input)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model layers based on provided code:
        # Input shape: (None,) - variable-length sequences of token ids
        # Embedding to 64 dims, then a Dense layer outputting size 5000 (vocab size)
        self.embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=64)
        self.dense = tf.keras.layers.Dense(5000)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.dense(x)  # Output shape: (B, T, 5000)
        return x

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Generate a random batch of variable length sequences to match model input.
    # Assumptions:
    # - batch size = 64 (default training batch size)
    # - sequence length random between 1 and 64 (max_length)
    # - values in [1, 5000) as token indices
    batch_size = 64
    max_length = 64

    import numpy as np

    # Create ragged tensor input with shape (batch_size, variable sequence length)
    # Since TF models expect dense tensors, we pad sequences for batching:
    seq_lengths = np.random.randint(1, max_length+1, size=batch_size)
    max_seq_len = seq_lengths.max()

    # Generate padded random token indices for each sequence
    input_array = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    for i, length in enumerate(seq_lengths):
        input_array[i, :length] = np.random.randint(1, 5000, size=length)
    return tf.constant(input_array, dtype=tf.int32)


# tf.random.uniform((B, None), dtype=tf.int32) ← input is batch of variable-length sequences of token ids

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self, vocabulary_size=20000, embedding_dim=128, hidden_units=100):
        super(MyModel, self).__init__()
        # Embedding layer with mask_zero=True generates a mask to ignore padding tokens (0)
        self.Embedding = tf.keras.layers.Embedding(
            vocabulary_size, embedding_dim, mask_zero=True
        )
        # Dense with relu activation
        self.FC1 = tf.keras.layers.Dense(hidden_units * 3, activation=tf.nn.relu)
        self.BN1 = tf.keras.layers.BatchNormalization()
        # ReLU layer (not wrapped in a way that loses mask; aware of the issue)
        self.RELU = tf.keras.layers.ReLU()
        self.GRU = tf.keras.layers.GRU(hidden_units, return_sequences=True)
        self.FC2 = tf.keras.layers.Dense(vocabulary_size)
        self.BN2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        # Inputs: tensor shape (batch_size, seq_len), integer token ids with padding=0
        x = self.Embedding(inputs)  # shape: (B, seq_len, embedding_dim)
        x = self.FC1(x)
        x = self.BN1(x, training=training)
        
        # Print mask before ReLU (for debug; in real usage remove prints)
        # Note: x._keras_mask holds the boolean mask tensor propagated from Embedding layer
        tf.print("before relu:", x._keras_mask)
        
        # Critical point: tf.keras.layers.ReLU() can drop _keras_mask due to internal tf.nn.relu usage.
        # Here we just apply ReLU and accept mask loss as per TF <= 2.1 behavior in the original issue.
        x = self.RELU(x)
        
        tf.print("after relu:", x._keras_mask)
        
        # Note: Because mask may be lost after ReLU, passing mask explicitly to GRU
        # if available, use mask from embedding layer to inform recurrent layer about padded tokens
        if mask is None:
            mask = self.Embedding.compute_mask(inputs)
        x = self.GRU(x, mask=mask, training=training)

        x = self.FC2(x)
        x = self.BN2(x, training=training)
        x = self.RELU(x)
        return x


def my_model_function():
    # Instantiate MyModel with default parameters matching the original example
    return MyModel()


def GetInput():
    # Generate random batch of padded sequences similar to the example
    # Batch size: 32, sequence length: variable up to 20, token ids between 1 and 19999 (0 is padding)
    batch_size = 32
    max_seq_len = 20
    vocabulary_size = 20000

    # Randomly generate sequences with lengths between 3 and 20, and pad zeros at end
    import numpy as np

    # Random sequence lengths
    seq_lengths = np.random.randint(3, max_seq_len + 1, size=(batch_size,))
    sequences = []
    for length in seq_lengths:
        # Generate random integers from 1 to vocabulary_size-1 to avoid zero (padding)
        seq = np.random.randint(1, vocabulary_size, size=(length,))
        # Pad sequence with zeros to max_seq_len
        if length < max_seq_len:
            seq = np.pad(seq, (0, max_seq_len - length), constant_values=0)
        sequences.append(seq)
    sequences = np.stack(sequences).astype(np.int32)  # shape (batch_size, max_seq_len)

    # Return a tensor matching the model input shape (B, None)
    return tf.convert_to_tensor(sequences)


# Notes/Assumptions:
# - The original issue showed that tf.keras.layers.ReLU drops the _keras_mask attribute,
#   which leads to incorrect masking behavior in downstream layers like GRU and loss calculation.
# - This code replicates the original model structure and behavior as closely as possible.
# - The ReLU mask drop is left as-is since it reflects the reported bug in TF 2.1 and earlier.
# - The input generated in GetInput() provides batches of padded sequences with mask_zero in embedding layer.
# - Print statements inside the call method are replaced with tf.print for compatibility with graph mode.


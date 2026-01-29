# tf.random.uniform((B, T), dtype=tf.int64) ← Input shape is (batch_size, variable_sequence_length), padded in dataset with shape [None, None]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding layer for vocabulary size 51 and output dimension 100
        self.emb = tf.keras.layers.Embedding(51, 100)
        # Dense layer projecting embeddings back to vocabulary size 51
        self.layer = tf.keras.layers.Dense(51)

    def call(self, x):
        x = self.emb(x)
        x = self.layer(x)
        return x


def my_model_function():
    # Return an instance of the MyModel. No pretrained weights specified.
    return MyModel()


def GetInput():
    # Generate a random batch of padded sequences matching the input expected by MyModel.
    # Assumptions:
    # - Batch size: 4 (same as the original padded_batch(4))
    # - Sequence length: random between 10 and 50, padded up to max length 50 for simplicity
    # - Integer values in [0, 50] as token IDs, dtype int64

    batch_size = 4
    max_seq_len = 50
    vocab_size = 51

    # Create a list of random sequence lengths between 10 and 50
    seq_lengths = tf.random.uniform(shape=(batch_size,), minval=10, maxval=max_seq_len + 1, dtype=tf.int32)

    # Initialize a tensor of zeros (padding token assumed 0)
    inputs = tf.zeros((batch_size, max_seq_len), dtype=tf.int64)

    # For each sequence, generate random tokens and pad with zeros
    inputs_list = []
    for i in range(batch_size):
        length = seq_lengths[i].numpy()
        # Generate random ints in [1,50] for tokens (excluding padding 0)
        tokens = tf.random.uniform((length,), minval=1, maxval=vocab_size, dtype=tf.int64)
        # Pad tokens to max_seq_len with zeros at the end
        padded = tf.pad(tokens, [[0, max_seq_len - length]])
        inputs_list.append(padded)

    # Stack into a tensor
    inputs = tf.stack(inputs_list, axis=0)
    return inputs


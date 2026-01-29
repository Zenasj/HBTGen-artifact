# tf.random.uniform((B, 5, 3), dtype=tf.float32) ← Input shape inferred from batching 5-length sequences one-hot encoded over 3 classes

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture as per example in the issue:
        # Input shape (5, 3) flattened to vector 15, then Dense(4) output
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        y = self.dense(x)
        return y

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of 5 samples,
    # each is a sequence length 5 with one-hot dimension 3 (like one-hot encoding of letters)
    # Value range is [0,1) floats - compatible since the actual input is one-hot vectors.
    batch_size = 5  # matching the batch size used in the example dataset batching
    seq_length = 5
    one_hot_dim = 3
    # The input expected is (batch_size, 5, 3) float tensor
    # Since the original example maps strings to one-hot encoded (5,3) sequences,
    # we generate random one-hot like inputs by sampling random indices and convert them to one-hot.
    indices = tf.random.uniform(shape=(batch_size, seq_length), maxval=one_hot_dim, dtype=tf.int32)
    one_hot_input = tf.one_hot(indices, depth=one_hot_dim, dtype=tf.float32)
    return one_hot_input


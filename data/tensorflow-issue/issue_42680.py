# tf.random.uniform((B, 1), dtype=tf.int64)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    """
    This model replicates the key setup described in the issue:
    - An integer input feature representing something like month IDs from 1 to 12 (integers not starting at 0).
    - A CategoryEncoding layer with output_mode="binary".
    
    The known issue was that adapting CategoryEncoding directly on integer lists
    not starting at zero results in an output shape that is max(input) + 1.
    The recommended fix is to specify num_tokens matching the cardinality of unique integers,
    or to use IntegerLookup + CategoryEncoding.

    Here, to demonstrate proper behavior, we:
    - Use IntegerLookup to convert arbitrary integer tokens to contiguous indices starting at 0.
    - Then use CategoryEncoding with num_tokens set correctly to the number of unique tokens.
    """

    def __init__(self):
        super().__init__()
        # From the issue: input integers range from 1 to 12
        self.unique_tokens = np.arange(1, 13)
        self.num_tokens = len(self.unique_tokens)  # 12

        # IntegerLookup is designed to handle inputs not starting at zero and map them to [0, num_tokens-1]
        self.lookup = tf.keras.layers.IntegerLookup(
            vocabulary=self.unique_tokens, mask_token=None, oov_token=None
        )
        # CategoryEncoding expecting indices from 0 to num_tokens-1
        self.encoding = tf.keras.layers.CategoryEncoding(
            num_tokens=self.num_tokens,
            output_mode="binary"
        )
        self.dense = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs):
        # inputs shape: (batch_size, 1), dtype=int64
        # Map input integers to contiguous indices
        indexed = self.lookup(inputs)
        # indexed shape: (batch_size, 1), each value in [0, num_tokens-1]

        # CategoryEncoding expects int inputs, shape (batch_size, 1)
        encoded = self.encoding(indexed)

        # encoded shape: (batch_size, num_tokens) → (None, 12)
        output = self.dense(encoded)
        return output

def my_model_function():
    """
    Return an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor of integers in the range 1..12 (inclusive), shape (batch_size, 1),
    dtype int64, matching expected input shape for MyModel.
    """
    batch_size = 4  # example batch size
    # Integers from 1 to 12 inclusive
    input_tensor = tf.random.uniform(
        shape=(batch_size, 1),
        minval=1,
        maxval=13,
        dtype=tf.int64
    )
    return input_tensor


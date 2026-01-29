# tf.random.uniform((B, 1), dtype=tf.int32) ← input shape inferred: single integer token per batch item for Embedding layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with vocab size 1 and embedding dim 1, input_length=1:
        # Corresponds to the example in the issue, so indices must be in [0,1).
        self.embedding = tf.keras.layers.Embedding(input_dim=1, output_dim=1, input_length=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

        # Secondary model (a simple Dense-only model without embedding)
        self.dense_only = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])

    def call(self, x, training=False):
        # x is expected to be a batch of integer indices for embedding path
        # Pass through embedding model path
        emb_out = self.embedding(x)
        emb_out = self.flatten(emb_out)
        emb_out = self.dense(emb_out)

        # For comparison, pass same input cast to float32 into dense-only model (we need float inputs)
        # Since dense_only expects float inputs, convert x to float and shape accordingly.
        # Flatten x to (batch_size, 1) and convert to float32
        x_float = tf.cast(tf.reshape(x, [-1, 1]), tf.float32)
        dense_out = self.dense_only(x_float)

        # Compare the two outputs: Here we produce a numeric difference (emb_out - dense_out)
        # This implements a fusion of the two models and outputs a combined result.
        diff = emb_out - dense_out

        # Return a dictionary of outputs for clarity (could also just return diff)
        return {
            'embedding_output': emb_out,
            'dense_only_output': dense_out,
            'difference': diff
        }

def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tensor of integer indices valid for the embedding layer (must be in [0, 1) due to vocab size=1)
    # Shape = (batch_size=4, sequence_length=1), arbitrary batch size chosen for example
    # Since embedding vocab is 1, valid id is only 0
    batch_size = 4
    input_seq_len = 1
    return tf.random.uniform(
        shape=(batch_size, input_seq_len),
        minval=0,
        maxval=1,
        dtype=tf.int32
    )


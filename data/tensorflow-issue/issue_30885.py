# tf.random.uniform((B,), dtype=tf.int32) ← Input shape inferred from dataset range; sequence of integers

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize the large embedding variable from a fixed numpy array
        # Assuming vocab size large enough for sequence indices, dense embedding dim 400
        self.vocab_size = 100  # Based on input_fn Dataset range(100)
        self.embedding_dim = 400
        # Large constant numpy array to simulate big embedding variable initialization
        self.emb_init_np = np.random.rand(3000000, self.embedding_dim).astype(np.float32)
        # For demonstration, picking a slice to avoid massive memory usage in this TF 2.x conversion
        # NOTE: Original estimation was 3 million by 400; here to make runnable, slice down:
        self.emb_init_np = self.emb_init_np[:self.vocab_size, :]  # simplify embedding table size
        
        # Create embedding variable initialized from numpy array, non-trainable as in original
        self.embedding_var = tf.Variable(initial_value=self.emb_init_np,
                                         trainable=False,
                                         name="big_embedding")

        # Dense layer corresponding to logits
        self.dense = tf.keras.layers.Dense(1000)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs expected to be a 1D tensor of sequence indices: shape (B, )
        # embedding_lookup expects indices into embedding_var
        emb = tf.nn.embedding_lookup(self.embedding_var, inputs)
        logits = self.dense(emb)
        predictions = tf.greater(logits, 0.0)
        return predictions


def my_model_function():
    # Return instance of MyModel with initialized embedding_var from numpy array
    return MyModel()


def GetInput():
    # Return a 1D tensor of integer indices matching input range for embedding lookup
    # Using batch size B=8 arbitrarily, indices in [0, vocab_size)
    B = 8
    # Use tf.range clipped by vocab_size to simulate example input batch
    input_tensor = tf.random.uniform(shape=(B,), minval=0, maxval=100, dtype=tf.int32)
    return input_tensor


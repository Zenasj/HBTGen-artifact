# tf.random.uniform((B, 1), dtype=tf.int32) ← inferred input shape: batch size B unknown and input is (1,) integer (index) per input

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from the shared code snippet:
        self.n_inputs = 5       # number of input indices
        self.n_items = 1000     # vocabulary size for embedding layer
        self.dim_embedding = 10 # embedding output dimension

        # Shared embedding layer across multiple inputs
        self.shared_embed = tf.keras.layers.Embedding(self.n_items, self.dim_embedding)

        # Dense output layer after concatenation
        self.dense = tf.keras.layers.Dense(1)

        # Flatten layer, reused for each embedded output
        self.flatten = tf.keras.layers.Flatten()

        # Concatenate layer
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs, training=False):
        # inputs is expected to be a list of n_inputs tensors, each shape (batch_size, 1)

        embeds = []
        for i in range(self.n_inputs):
            x = inputs[i]
            x_emb = self.shared_embed(x)        # Shape: (batch_size, 1, dim_embedding)
            x_flat = self.flatten(x_emb)        # Shape: (batch_size, dim_embedding)
            embeds.append(x_flat)

        # Concatenate all embeddings along feature axis
        concat_embeds = self.concat(embeds)     # Shape: (batch_size, n_inputs * dim_embedding)

        # Dense layer produces output scalar per batch instance
        output = self.dense(concat_embeds)      # Shape: (batch_size, 1)

        return output

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Create a list of n_inputs tensors with shape (batch_size, 1) and dtype int32
    # We'll use batch_size = 32 as a reasonable default
    batch_size = 32
    n_inputs = 5
    n_items = 1000

    inputs = []
    for _ in range(n_inputs):
        # Random integers in [0, n_items) with shape (batch_size, 1)
        inp = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=n_items, dtype=tf.int32)
        inputs.append(inp)

    return inputs


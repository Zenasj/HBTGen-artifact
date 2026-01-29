# tf.RaggedTensor input with outer dimension (batch), variable inner dimensions (ragged), and feature shape (10, 10, 3)
# The model in the issue accepts two ragged tensor inputs of shape (None, None, 10, 10, 3),
# which means ragged batch, then ragged second dim, then fixed spatial dims and channels.

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two ragged tensor inputs expected
        # We'll implement the model similar to basic_ragged_graph from the issue
        
        # A Lambda to sum over the two ragged dimensions (axis 1 and 2)
        self.sum_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=(1, 2)))
        # Dense layer after flattening concatenated sums
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=2, activation=None)
        
    def call(self, inputs, training=False):
        # inputs: tuple of two ragged tensors, each with shape (batch, ragged, ragged, 10, 10, 3)
        # For each input, apply sum_layer which reduces over ragged dims (1, 2),
        # resulting in shape (batch, 10, 10, 3)
        summed = [self.sum_layer(inp) for inp in inputs]
        # Concatenate along last axis (channels/features)
        concat = tf.concat(summed, axis=-1)  # shape: (batch, 10, 10, 6)
        flat = self.flatten(concat)          # shape: (batch, 10*10*6)
        logits = self.dense(flat)            # shape: (batch, 2)
        return logits

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a batch size of 4 for ragged inputs matching the model's expected input
    batch_size = 4
    # For simplicity, create synthetic data with ragged outer two dims
    
    # Create random ragged lengths for dims 1 and 2
    # For each sample in batch, randomly decide number of "outer ragged rows" and "inner ragged rows"
    outer_lengths = np.random.randint(low=1, high=5, size=batch_size)
    inner_lengths_per_outer = [np.random.randint(low=1, high=5, size=outer) for outer in outer_lengths]
    
    # Fixed shape for each innermost tile (10,10,3)
    tile_shape = (10, 10, 3)
    
    ragged_tensors = []
    for idx in range(2):  # two inputs
        samples = []
        for b in range(batch_size):
            # For each batch element, create variable ragged dimension 1 = outer_lengths[b]
            outer_len = outer_lengths[b]
            inner_raggeds = []
            for inner_len in inner_lengths_per_outer[b]:
                # Create a (inner_len, 10, 10, 3) tensor with random values
                inner_raggeds.append(np.random.uniform(size=(inner_len,) + tile_shape).astype(np.float32))
            # inner_raggeds is list of arrays with variable length first dims
            # concatenate along axis=0 (concatenate all inner ragged rows for this batch elem outer rows)
            if len(inner_raggeds) > 0:
                concat_inner = np.concatenate(inner_raggeds, axis=0)
            else:
                concat_inner = np.empty((0,) + tile_shape, dtype=np.float32)
            samples.append(concat_inner)
        # Now create ragged tensor from row_lengths
        # First level ragged dimension: len=samples (batch size)
        # Each sample is 2D ragged: outer ragged dim lengths = outer_lengths, inner ragged dim lengths = inner_lengths_per_outer
        # For simplicity, flatten intermediate ragged dims into a single ragged dimension
        # We'll approximate by concatenating all inner raggeds per batch element and providing total row lengths
        
        # Flatten samples into single 2D ragged tensor with row lengths = sum of inner ragged lengths per batch element
        flat_values = np.concatenate(samples, axis=0)
        # row_lengths first correspond to batch elements, need total rows per batch elem
        row_lengths = [sample.shape[0] for sample in samples]
        
        ragged_tensor = tf.RaggedTensor.from_row_lengths(flat_values, row_lengths)
        ragged_tensors.append(ragged_tensor)
    
    return tuple(ragged_tensors)


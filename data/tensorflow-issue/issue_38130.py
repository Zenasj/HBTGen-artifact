# tf.random.uniform((batch_size, 1), dtype=tf.int32) ← Inference: input is integer indices of shape (batch_size, 1)

import tensorflow as tf

batch_size = 2  # from original example
nb_item = 5
nb_hidden = 3

class ScoMatLayer(tf.keras.layers.Layer):
    def __init__(self, embd_layer):
        super().__init__()
        # Gather all embeddings for all possible indices once at init time.
        # Note: This tensor is fixed, but embedding weights can change during training.
        all_index = tf.range(nb_item, dtype=tf.int32)
        self.embd_layer = embd_layer
        # Store embedding weights tensor reference; gather full embedding matrix on each call
        # to ensure gradients flow properly through embedding weights.
        
    def call(self, inputs):
        # inputs: shape (batch_size, nb_hidden)
        # Get full embeddings (nb_item, nb_hidden) on each call to ensure updated weights.
        embd2 = self.embd_layer(tf.range(nb_item, dtype=tf.int32))  # shape (nb_item, nb_hidden)
        # Compute similarity matrix: (batch_size, nb_hidden) x (nb_item, nb_hidden)^T => (batch_size, nb_item)
        return tf.matmul(inputs, embd2, transpose_b=True)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with nb_item embeddings of dimension nb_hidden
        self.embd_layer = tf.keras.layers.Embedding(nb_item, nb_hidden)
        self.sco_mat_layer = ScoMatLayer(self.embd_layer)

    def call(self, inputs):
        # inputs: int tensor shape (batch_size, 1)
        # embed and reshape to (batch_size, nb_hidden)
        embd1 = tf.reshape(self.embd_layer(inputs), [-1, nb_hidden])
        # compute score matrix between embd1 and full embedding matrix
        sco_mat = self.sco_mat_layer(embd1)  # shape (batch_size, nb_item)
        return sco_mat

def my_model_function():
    # Return a fresh instance of MyModel.
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (batch_size, 1) with valid indices in [0, nb_item).
    return tf.random.uniform((batch_size, 1), minval=0, maxval=nb_item, dtype=tf.int32)

# Notes:
# - This model replicates the example from the issue, using subclassing rather than a Lambda layer.
# - The ScoMatLayer queries the full embedding matrix on each call to ensure gradients propagate correctly.
# - The input shape is (batch_size, 1), integer indices for embedding lookup.
# - Output shape is (batch_size, nb_item), the score matrix.
# - Compatible with TF 2.20.0 and can be JIT compiled as requested.


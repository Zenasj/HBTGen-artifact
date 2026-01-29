# tf.random.uniform((1, 400), dtype=tf.float32) ← input shape inferred as (batch=1, features=400)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a Dense layer instead of tf.get_variable which is TF1 style
        # This Dense layer acts as the weights matrix of shape (400,1)
        # No activation is used to mimic sparse_tensor_dense_matmul behavior
        self.dense = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        # Expecting inputs to be sparse tensor of shape (batch_size, 400)
        # Perform sparse-dense matmul similar to tf.sparse_tensor_dense_matmul
        # tf.keras.layers.Dense accepts dense inputs, so we manually apply matmul
        # to sparse tensors
        if isinstance(inputs, tf.SparseTensor):
            # inputs is sparse, so multiply manually
            # weights is self.dense.kernel shape (400,1)
            weights = self.dense.kernel
            outputs = tf.sparse.sparse_dense_matmul(inputs, weights)
        else:
            # fallback - dense input
            outputs = self.dense(inputs)
        return outputs


def my_model_function():
    # Return an instance of MyModel (untrained, random weights)
    return MyModel()


def GetInput():
    # Return a SparseTensor input with shape (1, 400) matching the model's input
    # Creating a sparse tensor with random indices and float values.

    # For demonstration, create a sparse tensor with 10 random nonzero values
    import numpy as np

    batch_size = 1
    feature_dim = 400
    nnz = 10  # number of nonzero elements

    # Random indices within shape (1,400)
    # Since batch is 1, indices are of form [batch_index, feature_index]
    indices = np.zeros((nnz, 2), dtype=np.int64)
    indices[:, 0] = 0  # batch index fixed to 0 for all
    indices[:, 1] = np.random.choice(feature_dim, nnz, replace=False)

    # Random float values for the entries
    values = np.random.rand(nnz).astype(np.float32)

    dense_shape = [batch_size, feature_dim]

    sparse_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    # It's generally good to reorder the sparse tensor indices for TF ops
    sparse_input = tf.sparse.reorder(sparse_input)

    return sparse_input


# tf.random.uniform((20, 100), dtype=tf.float32) ← inferred input shape from example usage ex_x shape=(20, 100)

import tensorflow as tf
import numpy as np

def matmul_dense_sparse(a, b):
    # Perform matmul with sparse second matrix 'b' and dense 'a'
    # Equivalent to a @ b but with b as sparse tensor.
    ta = tf.transpose(a)
    tb = tf.sparse.transpose(b)
    return tf.transpose(tf.sparse.sparse_dense_matmul(tb, ta))


class SparseLayer(tf.keras.layers.Layer):
    """
    Keras Layer with trainable weights representing the values of a SparseTensor matrix.

    This layer stores sparse indices and shape,
    and trainable weight vector corresponds to the values at those indices.

    The sparse matrix is formed by self.indices, self.w, self.shape,
    reordered once when weights change.

    Note: In eager execution, gradients are not properly propagated to SparseTensor values
    by default (as reported in the issue). This model implements the layer as in given example.
    """

    def __init__(self, indices, shape):
        super().__init__()
        # Store indices and sparse shape (fixed)
        # indices is array-like shape (num_nonzero, 2) for a 2D sparse matrix
        self.indices = tf.constant(indices, dtype=tf.int64)
        self.shape = tf.constant(shape, dtype=tf.int64)

    def build(self, input_shape):
        # Create a 1D trainable weight vector with length = number of nonzero elements
        self.w = self.add_weight(
            name='w',
            shape=(self.indices.shape[0],),
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        # Re-create sparse tensor with current weights as values
        sparse_mat = tf.sparse.reorder(tf.sparse.SparseTensor(self.indices, self.w, self.shape))
        # Compute matrix multiplication: inputs @ sparse matrix
        return matmul_dense_sparse(inputs, sparse_mat)


class MyModel(tf.keras.Model):
    """
    Model that uses the SparseLayer for linear transform with sparse weights.
    The input shape is (batch_size, 100) to match the sparse weight matrix shape (100, 5).
    """

    def __init__(self):
        super().__init__()
        # Sparse indices and shape defined as per the example in the issue
        indices = np.array([[1, 2], [30, 1], [30, 3], [45, 2], [56, 2], [32, 4]], dtype=np.int64)
        shape = (100, 5)
        self.sparse_layer = SparseLayer(indices, shape)

    def call(self, inputs):
        return self.sparse_layer(inputs)


def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()


def GetInput():
    # Return a random float32 input tensor shaped (20, 100) to match model input
    # Batch size 20 and feature size 100 matches example in the issue
    return tf.random.uniform((20, 100), dtype=tf.float32)


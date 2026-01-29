# tf.random.uniform((Nt,), dtype=tf.float32) or tf.random.uniform((Nt, D), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Emulates batched sparse matrix-vector/matrix multiplication with 
    very large sparse matrices that exceed int32 indexing limits on GPU.

    The sparse matrix is represented via row and column index arrays,
    and processed in batches using tf.tensor_scatter_nd_add to accumulate
    results without dense representation.

    This is inspired by the SparseBinaryMatrixWrapper from the issue,
    adapted as a tf.keras.Model subclass.
    """

    def __init__(self, Nc, Nt, row, col, batchsize=100000, **kwargs):
        """
        Args:
            Nc: int, number of output rows (height of sparse matrix)
            Nt: int, number of input columns (width of sparse matrix)
            row: 1D tf.Tensor[int64/int32], row indices of nonzero elements
            col: 1D tf.Tensor[int64/int32], column indices of nonzero elements
            batchsize: int, size of batches to process sparse indices
        """
        super().__init__(**kwargs)
        self.Nc = Nc
        self.Nt = Nt

        # Sparse indices should be int32 for GPU compatibility
        # If int64 provided, cast to int32 with warning
        if row.dtype != tf.int32:
            row = tf.cast(row, tf.int32)
        if col.dtype != tf.int32:
            col = tf.cast(col, tf.int32)

        self.row = row
        self.col = col
        self.batchsize = batchsize

        # Prepare batch boundaries on the sparse indices
        # Add len(row) as the final boundary, so parsing intervals of indices works
        num_indices = tf.shape(self.row)[0]
        boundaries = tf.range(0, num_indices, batchsize)
        # If last boundary is not exactly at end, append end index
        if boundaries[-1] != num_indices:
            boundaries = tf.concat([boundaries, [num_indices]], axis=0)
        self.batches = boundaries

    @tf.function
    def matvec(self, vec):
        """
        Multiply sparse matrix (Nc x Nt) by dense vector (Nt,)
        Returns: dense vector (Nc,)
        """
        result = tf.zeros((self.Nc,), dtype=vec.dtype)
        assert vec.shape[0] == self.Nt, "Input vector length must be Nt"
        num_batches = tf.shape(self.batches)[0] - 1

        for i in tf.range(num_batches):
            start = self.batches[i]
            end = self.batches[i + 1]

            rows_batch = self.row[start:end]
            cols_batch = self.col[start:end]

            vals = tf.gather(vec, cols_batch)  # shape: (batch_size,)
            # Accumulate vals to result at indices rows_batch
            result = tf.tensor_scatter_nd_add(result, tf.expand_dims(rows_batch, axis=1), vals)

        return result

    @tf.function
    def matmul(self, mat):
        """
        Multiply sparse matrix (Nc x Nt) by dense matrix (Nt x D)
        Returns: dense matrix (Nc x D)
        """
        shape = tf.shape(mat)
        assert shape[0] == self.Nt, "Input matrix shape mismatch, must have Nt rows"
        D = shape[1]
        result = tf.zeros((self.Nc, D), dtype=mat.dtype)
        num_batches = tf.shape(self.batches)[0] - 1

        for i in tf.range(num_batches):
            start = self.batches[i]
            end = self.batches[i + 1]

            rows_batch = self.row[start:end]
            cols_batch = self.col[start:end]

            vals = tf.gather(mat, cols_batch)  # shape: (batch_size, D)
            result = tf.tensor_scatter_nd_add(result, tf.expand_dims(rows_batch, axis=1), vals)

        return result

    @tf.function
    def matTmul(self, mat):
        """
        Multiply transpose of sparse matrix (Nt x Nc) by dense matrix (Nc x D)
        Returns: dense matrix (Nt x D)
        """
        shape = tf.shape(mat)
        assert shape[0] == self.Nc, "Input matrix shape mismatch, must have Nc rows"
        D = shape[1]
        result = tf.zeros((self.Nt, D), dtype=mat.dtype)
        num_batches = tf.shape(self.batches)[0] - 1

        for i in tf.range(num_batches):
            start = self.batches[i]
            end = self.batches[i + 1]

            rows_batch = self.row[start:end]  # rows in sparse matrix
            cols_batch = self.col[start:end]  # cols in sparse matrix

            vals = tf.gather(mat, rows_batch)  # shape (batch_size, D)
            # Accumulate vals into result at indices cols_batch (transposed)
            result = tf.tensor_scatter_nd_add(result, tf.expand_dims(cols_batch, axis=1), vals)

        return result

    @tf.function
    def matTvec(self, vec):
        """
        Multiply transpose of sparse matrix (Nt x Nc) by dense vector (Nc,)
        Returns: dense vector (Nt,)
        """
        result = tf.zeros((self.Nt,), dtype=vec.dtype)
        assert vec.shape[0] == self.Nc, "Input vector length must be Nc"
        num_batches = tf.shape(self.batches)[0] - 1

        for i in tf.range(num_batches):
            start = self.batches[i]
            end = self.batches[i + 1]

            rows_batch = self.row[start:end]
            cols_batch = self.col[start:end]

            vals = tf.gather(vec, rows_batch)  # shape (batch_size,)
            result = tf.tensor_scatter_nd_add(result, tf.expand_dims(cols_batch, axis=1), vals)

        return result


def my_model_function():
    """
    Returns a sample MyModel instance preloaded with dummy sparse matrix.
    This example uses a small sparse matrix for demonstration purposes,
    since the class is intended for very large sparse matrices.

    In practice, users instantiate with their actual large sparse indices.
    """
    import numpy as np

    Nc = 1000  # number of rows, e.g., output dimension
    Nt = 1200  # number of columns, e.g., input dimension
    batchsize = 100000  # large batch size for partitioning sparse indices (here irrelevant)

    # Create a random sparse pattern with ~1% density
    nnz = int(0.01 * Nc * Nt)
    # Random row and col indices for nonzero elements
    row_np = np.random.randint(0, Nc, size=(nnz,), dtype=np.int32)
    col_np = np.random.randint(0, Nt, size=(nnz,), dtype=np.int32)

    row = tf.convert_to_tensor(row_np, dtype=tf.int32)
    col = tf.convert_to_tensor(col_np, dtype=tf.int32)

    return MyModel(Nc=Nc, Nt=Nt, row=row, col=col, batchsize=batchsize)


def GetInput():
    """
    Generates a random dense vector input (shape=(Nt,)) compatible with the
    sparse matrix width Nt used in my_model_function instance.

    Note:
    Because Nt is fixed in my_model_function, this input shape must match.
    """
    # Assume Nt=1200 as in my_model_function
    Nt = 1200
    # Random float vector of length Nt
    return tf.random.uniform((Nt,), dtype=tf.float32)


# tf.SparseTensor with shape [3,4], tf.Tensor with shape [?, 3] → after sparse_dense_matmul and transpose output shape is (?, 4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer to process dense input from shape (8,) to (3,)
        self.dense_layer = tf.keras.layers.Dense(3, activation='relu')
        # Final output dense layer with sigmoid activation
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')

        # Generate a constant SparseTensor here as part of the model state
        # This replicates the fixed sparse matrix in the original example
        self.sparse_matrix = tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 2]],
            values=[1.0, 2.0],
            dense_shape=[3, 4]
        )

    def call(self, inputs):
        # inputs shape: (batch_size, 8)
        dense_out = self.dense_layer(inputs)  # (batch_size, 3)

        # Sparse-dense matmul with transposes as in original:
        # tf.sparse.sparse_dense_matmul(x[0], x[1], adjoint_a=True, adjoint_b=True)
        # Here x[0] = sparse_matrix, x[1] = dense_out

        # adjoint_a=True means transpose sparse_matrix: from [3,4] → [4,3]
        # adjoint_b=True means transpose dense_out: from [batch_size,3] → [3,batch_size]
        # sparse_dense_matmul([4,3], [3,b]) → [4, b]
        # Then we transpose output to (b,4)

        sp = self.sparse_matrix  # shape [3,4]
        # Transpose sparse a dims: [3,4] → [4,3]
        sp_t = tf.sparse.transpose(sp)  # shape [4,3]

        # Transpose dense_out from (b,3) → (3,b)
        dense_out_t = tf.transpose(dense_out)  # shape [3,b]

        # Sparse-dense matmul
        multiplied = tf.sparse.sparse_dense_matmul(sp_t, dense_out_t)  # shape [4,b]

        # Transpose back to (b,4)
        multiplied_t = tf.transpose(multiplied)  # shape [b,4]

        # Pass through final dense layer
        output = self.out_layer(multiplied_t)  # shape [b,1], dtype float32

        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Input shape expected is (batch_size, 8), values can be random floats
    # Use batch size 2 for example
    B = 2
    H = 8
    return tf.random.uniform((B, H), dtype=tf.float32)


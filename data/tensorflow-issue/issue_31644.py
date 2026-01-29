# tf.random.uniform((B, 20), dtype=tf.float32) ← Input shape inferred from Input(shape=(20,)) in the provided code
import tensorflow as tf
import numpy as np
from scipy import sparse

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define Dense layers matching the original model
        self.dense1 = tf.keras.layers.Dense(20)
        self.dense2 = tf.keras.layers.Dense(30)

        # Prepare fixed sparse tensor inside the model initialization to avoid loading delays
        dense_mat = np.eye(30, 30, dtype=np.float32)
        sparse_mat = sparse.coo_matrix(dense_mat)
        sparse_indices = np.mat([sparse_mat.row, sparse_mat.col]).transpose()
        # Convert to tensor indices as int64 as required by SparseTensor
        sparse_indices_tf = tf.convert_to_tensor(sparse_indices, dtype=tf.int64)
        sparse_values_tf = tf.convert_to_tensor(sparse_mat.data, dtype=tf.float32)
        sparse_shape_tf = tf.convert_to_tensor(sparse_mat.shape, dtype=tf.int64)
        self.sparse_tensor = tf.SparseTensor(indices=sparse_indices_tf,
                                             values=sparse_values_tf,
                                             dense_shape=sparse_shape_tf)

    def call(self, inputs):
        # Forward logic: pass inputs through dense layers
        x = self.dense1(inputs)
        x = self.dense2(x)
        # Transpose x for sparse matmul operation as per original lambda logic
        x_t = tf.transpose(x)
        # Sparse-dense matmul with fixed sparse tensor
        y = tf.sparse.sparse_dense_matmul(self.sparse_tensor, x_t)
        # Transpose back to original orientation
        y_t = tf.transpose(y)
        return y_t

def my_model_function():
    # Returns a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (batch_size=1, 20)
    # Matching the Input(shape=(20,)) from original code
    return tf.random.uniform((1, 20), dtype=tf.float32)


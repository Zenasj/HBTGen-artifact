# tf.sparse.sparse_dense_matmul expects a SparseTensor and a dense tensor; input shapes inferred from example code:
# SparseTensor shape: (30, 30), dense tensor shape: (batch_size, 30)
# Overall input to model: dense input vector of shape (20,)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example:
        # Input shape: (20,)
        # Dense layers: Dense(20) -> Dense(30)
        self.dense1 = Dense(20)
        self.dense2 = Dense(30)
        # Naively embedding the sparse matrix as an instance variable (non-trainable)
        # Sparse matrix is fixed; to simulate, we create it inside the model for XLA compatibility.
        # We will embed the sparse matrix as a tf.SparseTensor constant.
        # In real scenario, this would come externally, but the original example hardcodes an eye matrix of 30x30.
        
        # Create the sparse tensor representing identity matrix 30 x 30
        indices = tf.constant([[i, i] for i in range(30)], dtype=tf.int64)  # shape (30,2)
        values = tf.constant([1.0]*30, dtype=tf.float32)
        dense_shape = tf.constant([30, 30], dtype=tf.int64)
        self.sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    def call(self, inputs):
        # inputs: shape (batch_size, 20)
        x = self.dense1(inputs)    # (batch_size, 20)
        x = self.dense2(x)         # (batch_size, 30)
        # The example applies sparse_dense_matmul with the sparse matrix multiplied by dense transposed x,
        # then transposed back.
        # sparse_dense_matmul uses shape (30,30) sparse tensor multiplied by (30, batch_size).
        # Our x has shape (batch_size, 30), so transpose to (30, batch_size).
        
        x_t = tf.transpose(x)      # shape (30, batch_size)
        
        # sparse_dense_matmul: (sparse 30x30) x dense (30 x batch_size) -> (30 x batch_size)
        y = tf.sparse.sparse_dense_matmul(self.sparse_tensor, x_t)
        
        # transpose back to (batch_size, 30)
        y = tf.transpose(y)
        
        return y

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a dense input tensor matching the expected input shape (batch_size, 20)
    # Using batch size 4 (reasonable default)
    # Random uniform input, float32
    return tf.random.uniform((4, 20), dtype=tf.float32)


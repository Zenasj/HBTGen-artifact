# tf.SparseTensor with indices shape (num_nonzero, 2), values shape (num_nonzero,), and dense_shape (2, 2) 
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, x, **kwargs):
        # `x` is expected to be a tf.SparseTensor
        # Return the sparse values directly to demonstrate sparse input handling
        return x.values

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tf.SparseTensor matching the expected input shape and type
    # Indices specify positions, values give the element value, and dense_shape is the shape of the sparse tensor
    indices = tf.constant([[0, 1], [1, 0]], dtype=tf.int64)
    values = tf.constant([1, 1], dtype=tf.int32)
    dense_shape = tf.constant([2, 2], dtype=tf.int64)
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    return sparse_tensor


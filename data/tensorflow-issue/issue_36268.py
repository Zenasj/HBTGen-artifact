# tf.sparse.SparseTensor with shape [3] (1D sparse tensor)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable layers needed: this model simply performs shape assertions.

    def call(self, inputs):
        # inputs is expected to be a tf.sparse.SparseTensor
        # For demonstration, perform shape assertion compatible with sparse tensor.
        # We do not convert sparse to dense to avoid OOM on large sparse tensors.
        
        # Since tf.debugging.assert_shapes does not support SparseTensor,
        # implement a custom shape check for sparse tensors:
        st = inputs
        # The shape property of SparseTensor is a tf.Tensor (int64 vector)
        shape = st.shape
        # Assert that shape is as expected: here assume 1D tensor with dimension 3
        # as per example in the issue.
        expected_shape = tf.constant([3], dtype=tf.int64)

        # We raise tf.errors.InvalidArgumentError if shape mismatches.
        # Use tf.debugging.assert_equal on shape vector.
        with tf.control_dependencies([
            tf.debugging.assert_equal(shape, expected_shape,
                message="SparseTensor shape mismatch")
        ]):
            # Return indices as an example output (could also return st itself)
            return st

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a SparseTensor input matching expected shape [3]
    dense = tf.constant([10, 20, 30], dtype=tf.int32)
    # Create sparse tensor from dense tensor
    st = tf.sparse.from_dense(dense)
    return st


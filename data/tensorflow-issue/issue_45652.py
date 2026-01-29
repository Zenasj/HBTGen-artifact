# tf.sparse.SparseTensor with shape (10, 10), and a dense tensor of shape (10, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Internal Sequential model with one Dense layer of 32 units,
        # similar to the "Crashes" example in the issue.
        self.fc = tf.keras.Sequential([tf.keras.layers.Dense(32)])

    def call(self, inputs):
        a, b = inputs  # a is SparseTensor, b is dense tensor
        b = self.fc(b)  # Apply sequential Dense layer to dense input b
        # Perform sparse-dense matrix multiplication
        result = tf.sparse.sparse_dense_matmul(a, b)
        return result

def my_model_function():
    # Create and build model with example input shapes
    model = MyModel()
    # We must build the model by running it once with a sample input to ensure weights are created
    sample_sparse = tf.sparse.from_dense(tf.ones((10, 10)))
    sample_dense = tf.ones((10, 10))
    model([sample_sparse, sample_dense])
    return model

def GetInput():
    # Return a tuple matching the inputs expected by MyModel:
    # - a SparseTensor of shape (10, 10)
    # - a dense Tensor of shape (10, 10)
    a = tf.sparse.from_dense(tf.ones((10, 10), dtype=tf.float32))
    b = tf.ones((10, 10), dtype=tf.float32)
    return (a, b)


# tf.random.uniform((B, H), dtype=tf.float32) ← Input shape inferred as 2D tensor matching example [[1.0, 2.0], [3.0, 4.0]] in provided code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        """
        Forward pass converts the dense input tensor to sparse,
        performs a sparse minimum with a sparse tensor of ones,
        then returns the resulting SparseTensor.

        Note: The original code returns a SparseTensor object. In a Keras context,
        returning SparseTensor directly is unusual because downstream ops expect dense tensors.
        But this faithfully reproduces the original behavior.
        
        This function fails to XLA compile due to unsupported sparse ops in XLA.
        """
        x_sparse = tf.sparse.from_dense(x)
        ones_sparse = tf.sparse.from_dense(tf.ones_like(x))
        x_min = tf.sparse.minimum(x_sparse, ones_sparse)
        return x_min

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a random dense float32 tensor matching the shape (2, 2) used in the example.
    # Assumed batch size and shape from example [ [1.0, 2.0], [3.0,4.0] ].
    shape = (2, 2)
    return tf.random.uniform(shape, dtype=tf.float32)


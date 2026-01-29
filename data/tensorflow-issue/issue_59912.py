# tf.random.uniform((64,), dtype=tf.float32) ← input shape inferred from original numpy 1D arrays of length 64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize a variable 'c' similar to the numpy array c=np.ones([64])
        self.c = tf.Variable(tf.ones([64], dtype=tf.float32), trainable=False)

    def assign(self, d):
        # Assign values of tensor d element-wise to variable self.c via tf.tensor_scatter_nd_update
        # instead of Python loop to keep graph compatibility
        # Create indices for update (0 to 63)
        indices = tf.reshape(tf.range(64), (-1, 1))
        self.c.assign(tf.tensor_scatter_nd_update(self.c, indices, d))

    def call(self, a, b):
        # Compute difference d = a - b
        d = a - b
        # Assign d value to self.c with TensorFlow ops (avoiding Python loops)
        self.assign(d)
        # Return the updated variable as output
        return self.c

def my_model_function():
    # Return an instance of MyModel with initial variable initialized to ones of shape [64]
    return MyModel()

def GetInput():
    # Return two tensors, each same shape and dtype as original numpy arrays (64,)
    # Using uniform random values as example inputs for subtraction
    a = tf.random.uniform((64,), dtype=tf.float32)
    b = tf.random.uniform((64,), dtype=tf.float32)
    return a, b


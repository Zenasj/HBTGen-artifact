# tf.random.uniform((B=1000, H=1000, W=20), dtype=tf.int64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model encapsulates two "modes":
        # - A native TF random tensor generation (simulating tensor creation inside TF)
        # - A fallback using tf.py_function to call a numpy function (simulating original test from the issue)
        # This allows demonstration/comparison of memory leak prone usage vs safe usage.

    def numpy_test(self, xyz_batch, k):
        # The original "test" function rewritten for tf.py_function usage:
        # Produces two arrays: indices (int64) and dist (float32) of shape (1000, 1000, 20).
        # This matches the original issue's example.
        import numpy as np
        indices = np.zeros((1000, 1000, 20), dtype=np.int64)
        dist = np.zeros((1000, 1000, 20), dtype=np.float32)
        return indices, dist

    def call(self, x):
        # x is a dummy input, ignored here, as original test function also had unused inputs
        # Output from two approaches:

        # 1. Using tf.random.uniform to produce a tensor (simulating a TF-based op, no leak)
        tf_indices = tf.random.uniform(shape=(1000, 1000, 20), minval=0, maxval=10, dtype=tf.int64)

        # 2. Using tf.py_function to call the numpy function (which had memory leak in TF 2.6, 2.7)
        # This call would return two tensors from numpy arrays.
        indices, dist = tf.py_function(
            func=lambda b, k: self.numpy_test(b, k),
            inp=[tf.constant(0), tf.constant(0)],
            Tout=[tf.int64, tf.float32]
        )
        # Set shapes for static shape inference
        indices.set_shape((1000, 1000, 20))
        dist.set_shape((1000, 1000, 20))

        # For demonstration: return a dict of the two outputs from different approaches
        # This highlights difference between native TF tensor creation and py_function output
        return {
            "tf_indices": tf_indices,  # safe TF tensor
            "pyfunc_indices": indices, # potentially memory-leaking in some TF versions
            "pyfunc_dist": dist
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return dummy input compatible with MyModel.call signature.
    # Since the input is unused by the model, provide a scalar int64 tensor.
    return tf.constant(0)


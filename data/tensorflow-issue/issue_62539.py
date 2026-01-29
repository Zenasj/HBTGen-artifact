# tf.random.uniform((20, 30, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed, purely functional ops

    @tf.function(jit_compile=True)
    def call(self, inp):
        """
        The fused model combines the logic of Model1 and Model2 as described:
          - Both compute softmax along axis=0 on inp
          - Both transpose softmax with perm=[0, 2, 1]
          - Both compute reduce_sum along axis=0 on transposed softmax
          - Both cast reduce_sum result to int32
          - Model2 additionally concatenates transposed softmax with itself along axis=1

        For fusion:
          - Compute all intermediate results once
          - Return tuple with:
            (reduce_sum, casted reduce_sum, concatenated transposed softmax)

        This matches Model2's return but includes Model1's outputs implicitly.
        """
        # Input expected shape: [20, 30, 1], dtype=tf.float32

        softmax = tf.nn.softmax(inp, axis=0)  # Shape [20, 30, 1]
        trans = tf.transpose(softmax, perm=[0, 2, 1])  # Shape [20, 1, 30]
        reduce_sum = tf.math.reduce_sum(trans, axis=0)  # Shape [1, 30]
        cast = tf.cast(reduce_sum, dtype=tf.int32)  # Shape [1, 30], int32
        concat = tf.concat([trans, trans], axis=1)  # Shape [20, 2, 30]

        # The original reported inconsistency is between outputs of Model1 and Model2.
        # Return all outputs so a user can compare them similarly.
        return reduce_sum, cast, concat

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input of shape [20, 30, 1], dtype float32
    return tf.random.uniform((20, 30, 1), dtype=tf.float32)


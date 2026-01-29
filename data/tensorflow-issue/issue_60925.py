# tf.random.uniform((2, 2, 1), dtype=tf.float32) ← input shape and dtype inferred from original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2, 1), dtype=tf.float32)])
    def call(self, x):
        # The original model unpacks the input tensor along axis 0 into two tensors of shape (2, 1, 1)
        # Then concatenates them along the last axis (-1).
        # This should produce shape (2, 2) with values [[1, 3], [2, 4]].

        # Use tf.raw_ops.Unpack to match original operation exactly
        unpack_op = tf.raw_ops.Unpack(value=x, num=2, axis=0)
        concat = tf.concat(unpack_op, axis=-1)
        return concat

def my_model_function():
    # Return an instance of the MyModel class
    return MyModel()

def GetInput():
    # Produce input tensor matching the example input:
    # shape (2,2,1), dtype float32, values arbitrary but for the sake of demonstration,
    # let's use uniform random values.
    return tf.random.uniform((2, 2, 1), dtype=tf.float32)


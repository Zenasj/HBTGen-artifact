# tf.random.uniform([], minval=0, maxval=255, dtype=tf.int32) and tf.random.uniform([9], minval=0, maxval=255, dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The core operation where LeftShift is performed.
        # Using tf.raw_ops.LeftShift as in the issue to represent the bitwise left shift.

    def call(self, inputs):
        # inputs is expected to be a dict with keys "x" (scalar int32) and "y" ([9] int32)
        # The original issue shows a discrepancy between eager and jit-compiled LeftShift results.
        x = inputs["x"]
        y = inputs["y"]
        # Perform raw_ops.LeftShift with switched arguments as in the issue:
        # raw_ops.LeftShift(y=x, x=y) means shift each element of y by scalar x
        result = tf.raw_ops.LeftShift(y=x, x=y)
        return result

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate input matching the original shapes and types used in the issue.
    # x: scalar int32 in [0, 255)
    tensor_x = tf.random.uniform([], minval=0, maxval=255, dtype=tf.int32)
    # y: vector int32 of shape [9] in [0, 255)
    tensor_y = tf.random.uniform([9], minval=0, maxval=255, dtype=tf.int32)
    return {"x": tensor_x, "y": tensor_y}


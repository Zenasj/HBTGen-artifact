# tf.random.uniform((None,), dtype=tf.int32) ← Based on example: inputs are scalar int32 tensors (batch dim None assumed)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model mimics the logic of the example `MyTrackable`'s test_func from the issue:
        # It takes two scalar int32 tensors, returns a dict with 'sum' and 'difference'.
        # We implement the test_func logic directly here.

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32),
                                  tf.TensorSpec(shape=None, dtype=tf.int32)])
    def call(self, first_arg, second_arg):
        result1 = first_arg + second_arg
        result2 = first_arg - second_arg
        # The original example had an unused constant "dead_code" node.
        # We omit that here as it is dead code.
        return {"sum": result1, "difference": result2}

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a tuple of inputs matching the input_signature of MyModel.call.
    # Assume scalar int32 inputs.
    # Batch dimension is not explicitly used in the example,
    # so producing scalar tensors is consistent.
    first = tf.random.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
    second = tf.random.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
    return (first, second)


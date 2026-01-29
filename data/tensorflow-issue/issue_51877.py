# tf.random.uniform((1,), dtype=tf.float32) ← inferred input shape is a single scalar value wrapped as shape (1,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters, this model just compares two scalar values with tolerance
        # We encapsulate the "expected" and "actual" scalar values as inputs in call()

    def call(self, inputs):
        # inputs is expected to be a tuple (expected, actual)
        # where expected: shape=(), a scalar tf.Tensor (float)
        # actual: shape=(1,), a tf.Tensor representing value as single element tensor, e.g. tf.random.uniform((1,))
        expected, actual = inputs

        # Since the issue was comparing a scalar to a single-item numpy array/tensor,
        # we extract the scalar from the shape-1 tensor actual[0]
        actual_scalar = actual[0]

        # We define a numeric difference for assertAlmostEqual emulation:
        diff = tf.abs(expected - actual_scalar)

        # Assume tolerance (places=2 digits) means absolute difference < 0.005
        tolerance = 5e-3

        # Boolean tensor indicating close enough
        result = diff < tolerance

        return result  # boolean tensor scalar indicating pass/fail of comparison

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a tuple of (expected_scalar, actual_single_element_tensor)
    # assumed float32 dtype

    # For demonstration, expected close to 0.93933054, actual close to 0.93933064 (mimicking the example from issue)
    expected = tf.constant(0.9393305437365417, dtype=tf.float32)
    actual = tf.random.uniform(shape=(1,), minval=0.93, maxval=0.95, dtype=tf.float32)

    # For consistency with example, we override actual with fixed value close to expected
    actual = tf.constant([0.93933064], dtype=tf.float32)

    return (expected, actual)


# tf.random.uniform(())  # Scalar inputs as float32 values are tested in original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, places=7):
        super().__init__()
        self.places = places

    def call(self, inputs):
        """
        Inputs: tuple of two scalar tensors a, b (tf.float32).

        Returns: A dictionary with:
          - 'diff': absolute difference between a and b
          - 'is_almost_equal': boolean scalar, True if a and b are almost equal up to self.places decimal places.
        
        Rationale: This model mimics the logic of assertAlmostEqual but compatible with tf.Tensor inputs,
        similar to the issue discussed, avoiding numpy method calls that cause errors.
        """
        a, b = inputs
        diff = tf.abs(a - b)
        # Compare rounded difference to zero using tf.round and places:
        # Equivalent to Python's: round(diff, places) == 0
        scale = tf.pow(10.0, tf.cast(self.places, tf.float32))
        rounded_diff = tf.round(diff * scale) / scale
        is_almost_equal = tf.equal(rounded_diff, 0.0)
        return {'diff': diff, 'is_almost_equal': is_almost_equal}

def my_model_function():
    # Provide a default model instance with places=7 (default places in assertAlmostEqual)
    return MyModel()

def GetInput():
    # Return a tuple of two scalar float32 tensors similar to failing test case inputs
    # (matching the original test scenario using tf.constant scalars)
    a = tf.constant(1.00002, dtype=tf.float32)
    b = tf.constant(1.00001, dtype=tf.float32)
    return (a, b)


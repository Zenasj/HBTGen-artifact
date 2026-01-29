# tf.random.uniform((1, 3, 3), dtype=tf.float32) ← Based on example input shape and dtype from the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Softmax along axis=0 as described in the issue
        self.softmax_axis0 = tf.keras.layers.Softmax(axis=0)
        # Softmax along default axis=-1 for comparison (usual case)
        self.softmax_last_axis = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs, training=False):
        # Compute softmax along axis=0 and axis=-1 for the same input
        softmax_0 = self.softmax_axis0(inputs)
        softmax_neg1 = self.softmax_last_axis(inputs)

        # According to the issue discussion,
        # softmax on axis=0 with shape (1,3,3) leads to outputs all ones,
        # since axis=0 has length 1 and exp(x)/sum(exp(x)) = 1 along that axis.

        # Compute a boolean tensor showing where the two outputs differ
        # (they differ extensively when axis=0 is used, since outputs are all ones)
        difference = tf.math.not_equal(tf.round(softmax_0 * 1e6), tf.round(softmax_neg1 * 1e6))
        # Using scaled rounding to 1e-6 for numerical tolerance comparison

        # Return a dictionary of outputs for clarity
        return {
            "softmax_axis0": softmax_0,
            "softmax_axis_minus1": softmax_neg1,
            "difference_bool": difference
        }

def my_model_function():
    return MyModel()

def GetInput():
    # Input tensor matching example from issue: shape (1,3,3), dtype float32
    # Values are random but similar scale to example in issue
    return tf.random.uniform((1, 3, 3), minval=0.0, maxval=1.0, dtype=tf.float32)


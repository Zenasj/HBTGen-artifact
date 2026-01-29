# tf.random.uniform((4, 2), dtype=tf.float32) ← inferred input shape and dtype from issue example

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

class MyModel(tf.keras.Model):
    """
    This model demonstrates the difference between outputs of math_ops.sqrt (tf.Tensor)
    and K.sqrt (Keras Tensor), and performs a numerical comparison between them.
    The forward method returns a boolean tensor indicating element-wise equality within a tolerance.
    """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        # Small epsilon added to denominator as in the example
        self.epsilon = epsilon

    def call(self, inputs):
        # inputs is a tf.Tensor of shape (4, 2) with dtype float32
        v_t = inputs
        m_t = tf.ones_like(v_t)  # Placeholder numerator tensor same shape as v_t
        lr_t = tf.constant(0.01, dtype=v_t.dtype)  # Dummy learning rate scalar

        # Compute sqrt using math_ops.sqrt (produces a tf.Tensor, eager, evaluated)
        sqrt_math = math_ops.sqrt(v_t)
        # Compute sqrt using K.sqrt (produces a Keras symbolic Tensor, deferred)
        sqrt_keras = K.sqrt(v_t)

        # Perform the analogous division operation from the example
        y_math = m_t / (sqrt_math + self.epsilon)
        y_keras = m_t / (sqrt_keras + self.epsilon)

        # To compare these, we need to evaluate both as tensors
        # In TF2, K.sqrt returns a symbolic tensor, but call runs eagerly,
        # so both are Tensor-like and can be converted to concrete tf.Tensor

        # Because K.sqrt returns a symbolic Tensor in graph mode,
        # but we are in eager mode here, it behaves similarly to tf.sqrt.
        # So this code runs eagerly and both are tf.Tensors.

        # Calculate element-wise absolute difference
        diff = tf.abs(y_math - y_keras)

        # Return boolean tensor of whether diff elements are close within tolerance (1e-6)
        return tf.less_equal(diff, 1e-6)


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random uniform input tensor of shape (4, 2) with float32 dtype,
    # matching the example inputs from the issue
    return tf.random.uniform((4, 2), minval=1e-4, maxval=1e-2, dtype=tf.float32)


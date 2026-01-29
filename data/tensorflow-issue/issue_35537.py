# tf.random.uniform((2, 4, 64, 64, 64), dtype=tf.float32) ← input shape and type from example

import tensorflow as tf
from tensorflow.keras import layers, activations

class MyModel(tf.keras.Model):
    """
    A combined model encapsulating two ways of applying softmax on inputs:
    - Using layers.Activation('softmax') which defaults to axis=-1 and accepts dtype
    - Using activations.softmax function which allows specifying axis but no dtype argument
    
    The forward pass produces a boolean tensor indicating elementwise equality
    of the two outputs, highlighting the observed differences in behavior.

    This fused model demonstrates the issue:
    layers.Activation('softmax', dtype='float32') applies softmax on last axis (-1) with float32 output,
    while activations.softmax inputs softmax on axis=1, but dtype can't be set directly.

    The comparison outputs False in most places due to axis mismatch.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # layers.Activation defaults axis=-1, with dtype float32 (mixed precision compatible)
        self.sfm1 = layers.Activation('softmax', dtype='float32')

    def call(self, inputs):
        # Apply softmax using layers.Activation - axis=-1 by default
        out1 = self.sfm1(inputs)

        # Apply softmax function specifying axis=1 but internally casting inputs to float32 for mixed precision support
        # This is the recommended workaround from the discussion
        cast_inputs = tf.dtypes.cast(inputs, dtype=tf.float32)
        out2 = activations.softmax(cast_inputs, axis=1)

        # Return boolean tensor showing elementwise equality (mostly False due to axis difference)
        comparison = tf.math.equal(out1, out2)
        return comparison


def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Generate input tensor matching the example shape (2,4,64,64,64) with float32 dtype
    # This is the same shape as used in the example: batch=2, channels=4 (channel first), H=64, W=64, D=64 (assuming 3D spatial)
    # Leaves the model behavior consistent with given use case
    return tf.random.uniform((2, 4, 64, 64, 64), dtype=tf.float32)


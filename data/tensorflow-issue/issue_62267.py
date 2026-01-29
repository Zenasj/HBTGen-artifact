# tf.random.normal((3, 2, 2, 3, 4), dtype=tf.float32) ← The input shape inferred from the issue (input "x" tensor)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # In this model, we encapsulate the operations:
        # - IgammaGradA: gradient of the lower incomplete gamma function w.r.t "a"
        # - DivNoNan: division that returns 0 where denominator is zero instead of NaN
        #
        # Based on the issue, the main source of discrepancy is due to
        # randomness inside the call and jit_compile differences.
        #
        # To reproduce correct and comparable behavior,
        # we accept x, y, z inputs externally so that repeated calls use same inputs.
        #
        # y and z correspond to the inputs for IgammaGradA.x and DivNoNan.x respectively.
        # The issue shows y to be shape [4] and z shape [1,1,1,1].
        #
        # We will implement forward pass:
        #   igamma_grad = IgammaGradA(a=x, x=y)
        #   div_nonan = DivNoNan(y=igamma_grad, x=z)
        # and output div_nonan
        #
        # This matches the reproducible example from the issue.

    @tf.function(jit_compile=True)
    def call(self, x, y, z):
        # Defensive casts and sanitizing NaNs or infinities not needed here,
        # assuming inputs are well formed random tensors as in GetInput

        igamma_grad = tf.raw_ops.IgammaGradA(a=x, x=y)
        div_res = tf.raw_ops.DivNoNan(y=igamma_grad, x=z)
        return div_res


def my_model_function():
    # Return an instance of the model.
    # No weights to load here.
    return MyModel()


def GetInput():
    # Generate inputs matching expected shapes:
    # x: [3, 2, 2, 3, 4], dtype float32 as per original input example
    # y: [4], dtype float32 as used in IgammaGradA op in example
    # z: [1, 1, 1, 1], dtype float32 as used in DivNoNan op in example

    x = tf.random.normal([3, 2, 2, 3, 4], dtype=tf.float32)
    y = tf.random.normal([4], dtype=tf.float32)
    z = tf.random.normal([1, 1, 1, 1], dtype=tf.float32)

    return (x, y, z)


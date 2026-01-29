# tf.random.uniform((1, 55, 3, 27), dtype=tf.int8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights or layers; purely functional computations.

    @tf.function(jit_compile=True)
    def call(self, inp):
        # inp: shape [1, 55, 3, 27], dtype=tf.int8

        # Model1 computations:
        squeeze1 = tf.squeeze(inp, axis=0)  # shape [55, 3, 27]
        mul1 = tf.multiply(squeeze1, squeeze1)
        abs1 = tf.abs(mul1)  # shape [55, 3, 27]

        # Model2 computations:
        squeeze2 = tf.squeeze(inp, axis=0)  # shape [55, 3, 27]
        transposed = tf.transpose(squeeze2, perm=[1, 0, 2])  # shape [3, 55, 27]
        trans_mul = tf.multiply(transposed, transposed)  # shape [3, 55, 27]
        mul2 = tf.transpose(trans_mul, perm=[1, 0, 2])  # back to [55, 3, 27]
        abs2 = tf.abs(mul2)  # shape [55, 3, 27]

        # Compare the outputs from Model1 and Model2:
        # Use tf.reduce_all with elementwise equality and abs difference within tolerance
        # Use tolerances as seen in the issue: rtol=0.001, atol=0.001
        # We'll compute boolean tensor for elements where both conditions hold:
        diff = tf.abs(tf.cast(abs1, tf.float32) - tf.cast(abs2, tf.float32))
        max_val = tf.maximum(tf.abs(tf.cast(abs1, tf.float32)), tf.abs(tf.cast(abs2, tf.float32)))
        within_tol = diff <= (0.001 * max_val + 0.001)
        equal = tf.reduce_all(within_tol)

        # Output a dictionary reporting results, for clarity:
        # Also include abs1 and abs2 outputs to observe values if needed.
        return {
            "model1_abs": abs1,
            "model2_abs": abs2,
            "model2_trans_mul": trans_mul,
            "are_equal_within_tol": equal,
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random int8 tensor matching model input shape: [1, 55, 3, 27]
    # Use uniform distribution between -128 and 127 to cover int8 range
    inp = tf.random.uniform(
        shape=[1, 55, 3, 27],
        minval=-128,
        maxval=128,
        dtype=tf.int32
    )
    return tf.cast(inp, tf.int8)


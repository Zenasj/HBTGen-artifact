# tf.random.uniform((37, 1, 1, 15, 36), dtype=tf.int8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers are explicitly needed given the example,
        # as all ops are functional style TF ops (multiply, abs, reduce_max, transpose, argmax)

    @tf.function(jit_compile=True)
    def call(self, inp):
        """
        Implements the combined logic of Model1 and Model2 from the issue,
        acting as two sub-models internally and comparing their outputs.
        
        inp shape: [37, 1, 1, 15, 36], dtype int8

        Model1 logic:
          mul = inp * inp
          reduce_max = max(abs(mul), axis=2)

        Model2 logic:
          trans1 = transpose(inp, perm=[1, 0, 2, 3, 4])
          trans_mul = trans1 * trans1
          mul = transpose(trans_mul, perm=[1, 0, 2, 3, 4])
          reduce_max_2 = max(abs(mul), axis=2)
          argmax = argmax(mul, axis=4)

        The output combines the reduce_max from Model1 and Model2,
        and also returns trans1 and argmax as Model2 does.

        The forward returns a tuple:
          (reduce_max, reduce_max_2, trans1, argmax)
          
        This allows checking for the mismatch issue as reported.
        """
        # Model1 branch
        mul1 = tf.multiply(inp, inp)
        abs1 = tf.abs(mul1)
        reduce_max_1 = tf.math.reduce_max(abs1, axis=2)

        # Model2 branch
        trans1 = tf.transpose(inp, perm=[1, 0, 2, 3, 4])
        trans_mul = tf.multiply(trans1, trans1)
        mul2 = tf.transpose(trans_mul, perm=[1, 0, 2, 3, 4])
        abs2 = tf.abs(mul2)
        reduce_max_2 = tf.math.reduce_max(abs2, axis=2)
        argmax = tf.math.argmax(mul2, axis=4)

        return reduce_max_1, reduce_max_2, trans1, argmax


def my_model_function():
    # Instantiate and return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor of shape [37, 1, 1, 15, 36] with int8 dtype,
    # matching the input expected by MyModel.
    # Use uniform distribution between -128 to 127 (full int8 range)
    input_tensor = tf.cast(
        tf.random.uniform(shape=[37, 1, 1, 15, 36], minval=-128, maxval=128, dtype=tf.int32),
        tf.int8)
    return input_tensor


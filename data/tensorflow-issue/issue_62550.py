import random
import tensorflow as tf
from tensorflow import keras

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)


        # Layers or other Keras model objects

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [17, 10, 42] : complex128
        trans = tf.transpose(inp, perm=[2, 1, 0])
        cast = tf.cast(trans, dtype=tf.int64)
        sliced = cast[(slice(None, None, None), slice(-1, 9223372036854775807, 1), slice(None, None, None))]
        min1 = tf.minimum(cast, sliced)
        min2 = tf.minimum(min1, min1)
        return min1, min2,

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)


        # Layers or other Keras model objects

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [17, 10, 42] : complex128
        trans = tf.transpose(inp, perm=[2, 1, 0])
        v6_0, v6_1 = tf.split(trans, 2, axis=0)
        cast = tf.cast(trans, dtype=tf.int64)
        sliced = cast[(slice(None, None, None), slice(-1, 9223372036854775807, 1), slice(None, None, None))]
        min1 = tf.minimum(cast, sliced)
        min2 = tf.minimum(min1, min1)
        return min1, min2, v6_0, v6_1

tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()
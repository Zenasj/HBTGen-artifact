import math

import tensorflow as tf
print(tf.__version__)
from keras import layers

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
        x = tf.pow(x, x)
        x = tf.math.log(x)
        # NOTE: tf.experimental.numpy.log2 will also output wrong result with XLA
        return x


def simple_diff():
    m = MyModule()
    x = tf.constant(
        -1.5, shape=[1], dtype=tf.float32,
    )
    with tf.device('/CPU:0'):
        tf.config.run_functions_eagerly(True)
        out = m(x)
        print(out) # RIGHT! tf.Tensor([nan], shape=(1,), dtype=float32)
        tf.config.run_functions_eagerly(False)
    
    with tf.device('/CPU:0'):
        out = m(x)
        print(out) # NOTE: WRONG! tf.Tensor([-0.8774437], shape=(1,), dtype=float32)


simple_diff()

import numpy as np
a = -2; b = 0.5

print(f"{np.log2(np.power(a, b)) = }")
print(f"{b * np.log2(abs(a)) = }")
"""
np.log2(np.power(a, b)) = nan
b * np.log2(abs(a)) = 0.5
"""

a = -2; b = 0.5
print(f"{np.log2(np.power(a, b)) = }") # nan
print(f"{b * np.log2(a) = }") # nan

a = -2; b = 2
print(f"{np.log2(np.power(a, b)) = }") # 2.0
print(f"{b * np.log2(a) = }") # nan
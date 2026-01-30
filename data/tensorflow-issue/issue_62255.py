import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      tensor = tf.random.normal([6, 10, 1, 1], dtype=tf.bfloat16)
      x = tf.raw_ops.DivNoNan(x=x, y=tensor)        
      return x

m = Network()
inp = {
    "x": tf.random.normal([1, 1], dtype=tf.bfloat16),
}

with tf.device('/GPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/GPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
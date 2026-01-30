import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.Pow(x=x, y=tf.random.uniform([4, 1], minval=0, maxval=1000000, dtype=tf.int32))        
      return x

m = Network()
inp = {
    "x": tf.random.uniform([], minval=-1000000, maxval=1000000, dtype=tf.int32),
}

with tf.device('/GPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/GPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
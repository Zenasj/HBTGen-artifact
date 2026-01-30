import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.AdjustContrastv2(images=x, contrast_factor=tf.random.normal([], dtype=tf.float32))               
      return x

m = Network()
inp = {
    "x": tf.random.normal([8, 8, 8], dtype=tf.float32),
    }

with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/CPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)

tf.debugging.assert_near(
          tf.reduce_sum(tf.cast(no_op_res, tf.float64)), 
          tf.reduce_sum(tf.cast(op_res, tf.float64)), 
          atol=0.00001, 
          rtol=0.00001
)
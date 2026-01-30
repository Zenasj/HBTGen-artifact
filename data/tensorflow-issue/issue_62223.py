import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.Cos(x=x, )        
      x = tf.raw_ops.LeakyRelu(features=x, alpha=9.456766920329814)        
      return x

m = Network()
inp = {
    "x": tf.random.normal([10,9,8,1,8,3,2], dtype=tf.bfloat16),
}

with tf.device('/GPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/GPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
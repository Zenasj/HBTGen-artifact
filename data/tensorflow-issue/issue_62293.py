import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.BatchMatMulV2(y=x, adj_x=False,adj_y=False,x=tf.random.normal([8, 3], dtype=tf.float32))        
      x = tf.raw_ops.Cos(x=x, )        
      return x

m = Network()
inp = {
    "x": tf.random.normal([8, 3, 8, 7, 3, 4], dtype=tf.float32),
}

with tf.device('/GPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/GPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
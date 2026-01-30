import random

import tensorflow as tf
import traceback
tf.random.set_seed(42)

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.BatchMatMulV2(y=x, adj_x=False,adj_y=False,x=tf.random.normal([10, 4], dtype=tf.float32))        
      return x

m = Network()
inp = {
    "x": tf.random.normal([10, 4, 9, 10, 4, 7], dtype=tf.float32),
}

with tf.device('/GPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/GPU:0'):
        op_res = m(**inp)
    print(tf.cast(no_op_res, tf.float64) - tf.cast(op_res, tf.float64))
    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
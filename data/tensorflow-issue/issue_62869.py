import random

import tensorflow as tf

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x, y):
      x = tf.raw_ops.LeftShift(y=x, x=y)        
      return x

m = Network()
tensor_x = tf.random.uniform([],minval=0,maxval=255,dtype=tf.int32)
tensor_y = tf.random.uniform([9],minval=0,maxval=255,dtype=tf.int32)
inp = {
    "x": tensor_x,
    "y": tensor_y
}

with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/CPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
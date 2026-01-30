import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      random_tensor = tf.random.uniform([],minval=0,maxval=255,dtype=tf.int32)
      int8_tensor = tf.dtypes.cast(random_tensor, tf.int8)
      x = tf.raw_ops.RightShift(y=x, x=int8_tensor)        
      return x


m = Network()
random_tensor = tf.random.uniform([4,1],minval=0,maxval=255,dtype=tf.int32)
int8_tensor = tf.dtypes.cast(random_tensor, tf.int8)
inp = {
    "x": int8_tensor,
}

with tf.device('/GPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/GPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.Relu6(features=x, )        
      x = tf.raw_ops.Sigmoid(x=x, )        
      return x

m = Network()
tensor = tf.random.normal([10, 9, 8], dtype=tf.bfloat16)
inp = {
    "x": tensor,
    }

with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/CPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
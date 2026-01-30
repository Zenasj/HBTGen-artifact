import random

import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      real_part = tf.random.normal([9, 8], dtype=tf.float32)
      imag_part = tf.random.normal([9, 8], dtype=tf.float32)
      tensor = tf.complex(real_part, imag_part)
      tensor = tf.cast(tensor,dtype=tf.complex64)
      x = tf.raw_ops.ReciprocalGrad(dy=x, y=tensor)      
      return x

m = Network()
real_part = tf.random.normal([9, 8], dtype=tf.float32)
imag_part = tf.random.normal([9, 8], dtype=tf.float32)
tensor = tf.complex(real_part, imag_part)
tensor = tf.cast(tensor,dtype=tf.complex64)
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
import tensorflow as tf
import traceback

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      
      x = tf.raw_ops.Acos(x=x, )        
      x = tf.raw_ops.Exp(x=x, )        
      return x

m = Network()
dic = {'ele': (-731778.6211090556-59304.1731637927j), 'size': [], 'dtype': tf.complex128}
inp = {
    "x": tf.constant(dic['ele'], dtype=tf.as_dtype(dic['dtype'])),
}

with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/CPU:0'):
        op_res = m(**inp)

    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
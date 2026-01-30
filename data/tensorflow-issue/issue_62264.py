import math
import random

import tensorflow as tf
import traceback

def replace_special_values(tensor):
    # Convert tensor to tf.float32 if it's not a supported dtype
    supported_dtypes = [tf.float16, tf.float32, tf.float64, tf.bfloat16]
    if tensor.dtype not in supported_dtypes:
        original_dtype = tensor.dtype
        tensor = tf.cast(tensor, tf.float32)
    else :
        original_dtype = None
    
    # Replace NaNs with zeros
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    
    # Replace positive infinities with a large number (e.g., 1e30)
    tensor = tf.where(tf.math.is_inf(tensor), 100, tensor)
    
    # Replace negative infinities with a small number (e.g., -1e30)
    tensor = tf.where(tf.math.is_inf(tensor) & tf.math.less(tensor, 0), -100, tensor)
    
    # Convert tensor back to its original dtype
    if original_dtype is not None :
        tensor = tf.cast(tensor, original_dtype)
    return tensor

class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
      random_tensor = tf.random.uniform([9],minval=0,maxval=255,dtype=tf.int32)
      int8_tensor = tf.dtypes.cast(random_tensor, tf.int8)
      x = tf.raw_ops.LeftShift(y=x, x=int8_tensor)        
      return x

m = Network()
random_tensor = tf.random.uniform([],minval=0,maxval=255,dtype=tf.int32)
int8_tensor = tf.dtypes.cast(random_tensor, tf.int8)
inp = {
    "x": int8_tensor,
}

with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    no_op_res = m(**inp)
    tf.config.run_functions_eagerly(False)
    with tf.device('/CPU:0'):
        op_res = m(**inp)
    no_op_res = replace_special_values(no_op_res)
    op_res = replace_special_values(op_res)
    tf.debugging.assert_near(tf.cast(no_op_res, tf.float64), tf.cast(op_res, tf.float64), atol=0.001, rtol=0.001)
import math

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
      dic = {'ele': [[-921600.0, -499712.0, 638976.0, 344064.0]], 'size': [1, 4], 'dtype': tf.bfloat16}
      x = tf.raw_ops.DivNoNan(y=x, x=tf.constant(dic['ele'], dtype=tf.as_dtype(dic['dtype'])))       
      x = tf.raw_ops.Asin(x=x, )          
      return x

m = Network()
dic = {'ele': [[688128.0, -344064.0, -778240.0, -532480.0], [-466944.0, -888832.0, 843776.0, -172032.0], [-704512.0, -389120.0, -827392.0, -188416.0], [778240.0, 32768.0, 794624.0, 761856.0], [-122880.0, -110592.0, -811008.0, 139264.0], [-561152.0, 106496.0, -389120.0, -548864.0], [909312.0, -94208.0, -499712.0, 811008.0]], 'size': [7, 4], 'dtype': tf.bfloat16}

inp = {
    "x": tf.constant(dic['ele'], dtype=tf.as_dtype(dic['dtype'])),
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
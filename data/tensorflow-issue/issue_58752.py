import math
import random

import tensorflow as tf
import numpy as np

def main_jit():
  class my_mod(tf.Module):
    def __init__(self,name=None):
      super().__init__(name=name)
    @tf.function(jit_compile=True)
    def __call__(self):
      Input_0 = tf.random.stateless_uniform([2,],seed=(1, 2) ,minval=-5, maxval=5, dtype=tf.float64)
      cosh3 = tf.math.cosh(Input_0)
      cosh6 = tf.math.cosh(cosh3)
      tan11 = tf.math.tan(cosh6)
      return tan11
  mod = my_mod()
  res = mod()
  return res

def main():
  class my_mod(tf.Module):
    def __init__(self,name=None):
      super().__init__(name=name)
    def __call__(self):
      Input_0 = tf.random.stateless_uniform([2,],seed=(1, 2) ,minval=-5, maxval=5, dtype=tf.float64)
      cosh3 = tf.math.cosh(Input_0)
      cosh6 = tf.math.cosh(cosh3)
      tan11 = tf.math.tan(cosh6)
      return tan11
  mod = my_mod()
  res = mod()
  return res

res_jit = main_jit()
res = main()

np_res = np.allclose(res_jit.numpy(),res.numpy(),rtol=1e-1, atol=1e-3, equal_nan=True)
print(res_jit.numpy()-res.numpy())
print(np_res)
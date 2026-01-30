import math
import random

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops

class MyModule(tf.Module):
  def __init__(self):
    self.v = None

  @tf.function
  def __call__(self, x):
    if self.v is None:
      # 4 GiB variable
      self.v =  tf.Variable(tf.random.uniform((1024,1024,1024), dtype=tf.dtypes.float32))
    x = tf.math.reduce_sum(x * self.v)
    return array_ops.identity(x, name="output_0")

input = np.random.uniform(size=(1024,)).astype(np.float32)
func = MyModule()
out = func(input)

cfunc = func.__call__.get_concrete_function(tf.TensorSpec(input.shape, tf.float32))
tf.saved_model.save(func, 'my_saved_model', signatures=cfunc)
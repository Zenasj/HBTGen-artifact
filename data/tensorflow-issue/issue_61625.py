import random
from tensorflow.keras import layers

import tensorflow as tf
import os
import numpy as np
try:
  padding_0_0 = 125091515651
  padding_0_1 = 125091515651
  padding_0 = [padding_0_0,padding_0_1,]
  padding_1_0 = 125091515651
  padding_1_1 = 125091515651
  padding_1 = [padding_1_0,padding_1_1,]
  padding = [padding_0,padding_1,]
  data_format = None
  arg_class = tf.compat.v1.keras.layers.ZeroPadding2D(padding=padding,data_format=data_format,)
  arg_input_0_tensor = tf.random.uniform([3, 14, 14, 576], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))

### Relevant log output


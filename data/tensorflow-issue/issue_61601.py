import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os
import numpy as np
try:
  arg_0_0 = 1e+20
  arg_0_1 = True
  arg_0 = [arg_0_0,arg_0_1,]
  strides_0 = 2
  strides_1 = 2
  strides = [strides_0,strides_1,]
  arg_class = tf.keras.layers.MaxPooling2D(arg_0,strides=strides,)
  arg_input_0_tensor = tf.random.uniform([2, 17, 17, 768], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))

### Relevant log output


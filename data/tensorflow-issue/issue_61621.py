import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os
import numpy as np
try:
  padding = 1610612736
  arg_class = tf.keras.layers.ZeroPadding3D(padding=padding,)
  arg_input_0_tensor = tf.random.uniform([1, 1, 2, 2, 3], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))

### Relevant log output


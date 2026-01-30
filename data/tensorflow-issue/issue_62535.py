import math
import random
from tensorflow import keras

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        cos = tf.cos(inp)
        transpose = tf.transpose(cos, perm=[4, 1, 2, 3, 0])
        reduce_min = tf.math.reduce_min(transpose, axis=2)
        lrn = tf.raw_ops.LRN(input=reduce_min, depth_radius=1, bias=62.98211185437273, alpha=22.83989611654185, beta=0.9124946866870809)
        return lrn,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        cos = tf.cos(inp)
        transpose = tf.transpose(cos, perm=[4, 1, 2, 3, 0])
        reduce_min = tf.math.reduce_min(transpose, axis=2) # tf.squeeze(transpose, axis=3)
        lrn = tf.raw_ops.LRN(input=reduce_min, depth_radius=1, bias=62.98211185437273, alpha=22.83989611654185, beta=0.9124946866870809)
        return lrn, transpose,

inputs = [
tf.random.uniform(shape=[21, 27, 10, 1, 1], dtype=tf.float32),
]
model1 = Model1()
model2 = Model2()
device = "gpu"
with tf.device(device):
    tf.config.run_functions_eagerly(True)
    out1 = model1(*inputs)
    out2 = model2(*inputs)
    print(f'=========eager_output(version:{tf.__version__})================')
    try :
        for i in range(min(len(out1),len(out2))):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.001, atol=0.001, err_msg=f'at checking {i}th')
        print("XLA_eager does not trigger assertion")
    except AssertionError as e:
        print("XLA_eager triggers assertion")
        print(e)
    tf.config.run_functions_eagerly(False)
    out1 = model1(*inputs)
    out2 = model2(*inputs)
    print(f'=========compiled_output(version:{tf.__version__})================')
    try :
        for i in range(min(len(out1),len(out2))):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.001, atol=0.001, err_msg=f'at checking {i}th')
        print("XLA_complie does not trigger assertion")
    except AssertionError as e:
        print("XLA_complie triggers assertion")
        print(e)
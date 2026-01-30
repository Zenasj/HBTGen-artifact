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
    def __call__(self, inp1, inp2):
        trans = tf.transpose(inp1, perm=[1, 0])
        gather = tf.gather(trans, tf.clip_by_value(inp2, 0, 63), axis=0)
        squeeze = tf.squeeze(gather, axis=1) # replace with tf.math.reduce_min also trigger the error
        mul1 = tf.multiply(squeeze, squeeze)
        mul2 = tf.multiply(trans, mul1)
        return mul2,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        # Forward pass logic using TensorFlow operations
        # inp1: [64, 64] : complex128
        # inp2: [1, 1, 64] : int32
        trans = tf.transpose(inp1, perm=[1, 0])
        gather = tf.gather(trans, tf.clip_by_value(inp2, 0, 63), axis=0)
        squeeze = tf.squeeze(gather, axis=1) # replace with tf.math.reduce_min also trigger the error
        mul1 = tf.multiply(squeeze, squeeze)
        mul2 = tf.multiply(mul1, trans)
        return mul2, 

inputs = [
tf.random.uniform([64, 64], dtype=tf.float64),
tf.random.uniform(shape=[1, 1, 64], minval=-100, maxval=100, dtype=tf.int32),
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

tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()
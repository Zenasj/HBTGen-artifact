import random
from tensorflow import keras

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
params = [
]
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        trans = tf.transpose(inp, perm=[1, 0])
        flo = tf.floor(trans)
        rev = tf.reverse(trans, axis=[0])
        add = tf.add(rev, flo)
        return add,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        concat = tf.concat([inp, inp], axis=0)
        trans = tf.transpose(inp, perm=[1, 0])
        flo = tf.floor(trans)
        rev = tf.reverse(trans, axis=[0])
        add = tf.add(flo, rev)
        return add, concat, #v5_0,

inputs = [
tf.random.uniform(shape=[20, 20], dtype=tf.float64),
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
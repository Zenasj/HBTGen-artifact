import math
import random
from tensorflow import keras

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np

params = [
tf.random.uniform(shape=[49, 9, 1], dtype=tf.float32),
]
class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)
        self.p0 = tf.constant(params[0]) # [49, 9, 1] float32

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [7, 5, 49, 1, 1] : float32
        div = tf.divide(inp, self.p0)
        transposed_div = tf.transpose(div, perm=[0, 1, 3, 2, 4])
        red = tf.math.reduce_prod(transposed_div, axis=2)
        cos = tf.cos(red)
        return red, cos,

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)
        self.p0 = tf.constant(params[0]) # [49, 9, 1] float32

        # Layers or other Keras model objects

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [7, 5, 49, 1, 1] : float32
        div = tf.divide(inp, self.p0)
        transposed_div = tf.transpose(div, perm=[0, 1, 3, 2, 4])
        concat = tf.concat([transposed_div, transposed_div], axis=2)
        transposed_concat = tf.transpose(concat, perm=[1, 0, 2, 3, 4])
        red = tf.math.reduce_prod(transposed_div, axis=2)
        cos = tf.cos(red)
        return red, cos, transposed_concat

inputs = [
tf.random.uniform(shape=[7, 5, 49, 1, 1], dtype=tf.float32),
]
model1 = Model1()
model2 = Model2()
device = "cpu"
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
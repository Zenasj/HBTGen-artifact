import math
import random
from tensorflow import keras

import tensorflow as tf
import numpy as np
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        # Forward pass logic using TensorFlow operations
        # inp1: [13, 1] : int8
        # inp2: [13, 60] : int8
        add = tf.add(inp2, inp1)
        triu = tf.experimental.numpy.triu(add, k=0)
        reduce_max = tf.math.reduce_max(triu, axis=1)
        return triu, reduce_max, 

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        # Forward pass logic using TensorFlow operations
        # inp1: [13, 1] : int8
        # inp2: [13, 60] : int8
        add = tf.add(inp2, inp1)
        transpose = tf.transpose(add, perm=[1, 0])
        triu = tf.experimental.numpy.triu(add, k=0)
        reduce_max = tf.math.reduce_max(triu, axis=1)
        return triu, reduce_max, transpose

inputs = [
tf.cast(tf.random.uniform(shape=[13, 1], minval=-128, maxval=128, dtype=tf.int32), tf.int8),
tf.cast(tf.random.uniform(shape=[13, 60], minval=-128, maxval=128, dtype=tf.int32), tf.int8),
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
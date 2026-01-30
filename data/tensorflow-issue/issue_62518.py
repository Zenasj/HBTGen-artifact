import math
import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        mul = tf.matmul(inp2, inp1)
        rev = tf.reverse(mul, axis=[1])
        add = tf.add(rev, mul)
        argmax = tf.math.argmax(add, axis=1)
        return add, argmax,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        mul = tf.matmul(inp2, inp1)
        rev = tf.reverse(mul, axis=[1])
        add = tf.add(mul, rev)
        # add = tf.add(rev, mul) <- this will not trigger error
        argmax = tf.math.argmax(add, axis=1)
        return add, argmax,

inputs = [
tf.random.uniform(shape=[1, 18], dtype=tf.float32),
tf.random.uniform(shape=[34, 1], dtype=tf.float32),
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
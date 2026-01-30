import random
from tensorflow import keras

import tensorflow as tf
import numpy as np
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        trans = tf.transpose(inp, perm=[0, 1, 3, 2])
        round = tf.round(trans)
        greater = tf.greater(tf.reverse(round, axis=[0, 2]), round)
        logical_and = tf.logical_and(greater, greater)
        return greater, logical_and, 

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        trans = tf.transpose(inp, perm=[0, 1, 3, 2])
        round = tf.round(trans)
        greater = tf.greater(tf.reverse(round, axis=[0, 2]), round)
        logical_and = tf.logical_and(greater, greater)
        return greater, logical_and, trans,

inputs = [
tf.random.uniform(shape=[15, 1, 50, 35], dtype=tf.float64),
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
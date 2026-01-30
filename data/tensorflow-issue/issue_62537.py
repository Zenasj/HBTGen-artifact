import random
from tensorflow import keras

import tensorflow as tf
import numpy as np
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2, inp3, inp4, inp5):
        concat = tf.concat([inp4, inp3, inp2, inp1], axis=0)
        matmul = tf.matmul(inp5, concat)
        out = tf.transpose(matmul, perm=[1, 0])
        extra_trans = tf.transpose(matmul, perm=[1, 0])
        return out, extra_trans,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2, inp3, inp4, inp5):
        concat = tf.concat([inp4, inp3, inp2, inp1], axis=0)
        matmul = tf.matmul(inp5, concat)
        out = tf.transpose(matmul, perm=[1, 0])
        return out, 

inputs = [
tf.random.uniform(shape=[1, 8], dtype=tf.float64),
tf.random.uniform(shape=[1, 8], dtype=tf.float64),
tf.random.uniform(shape=[1, 8], dtype=tf.float64),
tf.random.uniform(shape=[1, 8], dtype=tf.float64),
tf.random.uniform(shape=[61, 4], dtype=tf.float64),
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
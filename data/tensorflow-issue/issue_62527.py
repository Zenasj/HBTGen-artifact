import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)
    @tf.function(jit_compile=True)
    def __call__(self, inp, inp2):
        concat = tf.concat([inp, inp2], axis=1)
        sliced = concat[:, -17:17:4]
        matmul = tf.matmul(sliced, sliced)
        return matmul,

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
    @tf.function(jit_compile=True)
    def __call__(self, inp, inp2):
        concat = tf.concat([inp, inp2], axis=1)
        transposed = tf.transpose(concat, perm=[1, 0])
        sliced = concat[:, -17:17:4]
        matmul = tf.matmul(sliced, sliced)
        return matmul, transposed,

inputs = [
tf.random.uniform(shape=[5, 1], dtype=tf.float16),
tf.random.uniform(shape=[5, 16], dtype=tf.float16),

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
import random
from tensorflow import keras

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np

###updated at 23.12.06
p0 = tf.random.uniform(shape=[6, 21, 59, 6], dtype=tf.float32)
p1 = tf.random.uniform(shape=[1, 54, 6, 6], dtype=tf.float32)

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
###updated at 23.12.06
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        conv2 = tf.nn.conv2d(self.p0, inp, strides=1, padding="SAME", dilations=(3, 3))
        _tan = tf.tan(conv2)
        return conv2, _tan

inputs = [
tf.random.uniform(shape=[18, 54, 6, 6], dtype=tf.float32),
]
model1 = Model1()
device = "gpu"
with tf.device(device):
    tf.config.run_functions_eagerly(True)
    out1 = model1(*inputs)
    out2 = model1(*inputs)
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
    out2 = model1(*inputs)
    print(f'=========compiled_output(version:{tf.__version__})================')
    try :
        for i in range(min(len(out1),len(out2))):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.001, atol=0.001, err_msg=f'at checking {i}th')
        print("XLA_complie does not trigger assertion")
    except AssertionError as e:
        print("XLA_complie triggers assertion")
        print(e)
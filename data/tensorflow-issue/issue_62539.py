import math
import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [28, 60, 1] : float64
        softmax = tf.nn.softmax(inp, axis=0)
        trans = tf.transpose(softmax, perm=[0, 2, 1])
        reduce_sum = tf.math.reduce_sum(trans, axis=0)
        cast = tf.cast(reduce_sum, dtype=tf.int32)
        return reduce_sum, cast,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [28, 60, 1] : float64
        softmax = tf.nn.softmax(inp, axis=0)
        trans = tf.transpose(softmax, perm=[0, 2, 1])
        concat = tf.concat([trans, trans], axis=1)
        reduce_sum = tf.math.reduce_sum(trans, axis=0)
        cast = tf.cast(reduce_sum, dtype=tf.int32)
        return reduce_sum, cast, concat,

inputs = [
tf.random.uniform(shape=[20, 30, 1], dtype=tf.float32),
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
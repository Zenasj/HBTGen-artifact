import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [1, 55, 3, 27] : int8
        squeeze = tf.squeeze(inp, axis=0)
        mul = tf.multiply(squeeze, squeeze)
        abs = tf.abs(mul)
        return abs,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [1, 55, 3, 27] : int8
        squeeze = tf.squeeze(inp, axis=0)
        transposed = tf.transpose(squeeze, perm=[1, 0, 2])
        trans_mul = tf.multiply(transposed, transposed)
        mul = tf.transpose(trans_mul, perm=[1, 0, 2])
        abs = tf.abs(mul)
        return abs, trans_mul,

inputs = [
tf.cast(tf.random.uniform(shape=[1, 55, 3, 27], minval=-128, maxval=128, dtype=tf.int32), tf.int8),
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
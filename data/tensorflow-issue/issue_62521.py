import math
import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [37, 1, 1, 15, 36] : int8
        mul = tf.multiply(inp, inp)
        abs = tf.abs(mul)
        reduce_max = tf.math.reduce_max(abs, axis=2)
        return reduce_max, 

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [37, 1, 1, 15, 36] : int8
        trans1 = tf.transpose(inp, perm=[1, 0, 2, 3, 4])
        trans_mul = tf.multiply(trans1, trans1)
        mul = tf.transpose(trans_mul, perm=[1, 0, 2, 3, 4])
        abs = tf.abs(mul)
        reduce_max = tf.math.reduce_max(abs, axis=2)
        argmax = tf.math.argmax(mul, axis=4)
        return reduce_max, trans1, argmax, # removing eitheor trans1 or argmax will not trigger error!

inputs = [
tf.cast(tf.random.uniform(shape=[37, 1, 1, 15, 36], minval=-128, maxval=128, dtype=tf.int32), tf.int8),
]
model1 = Model1()
model2 = Model2()
device = "cpu" # "gpu" also trigger the error
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

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [37, 1, 1, 15, 36] : int8
        mul = tf.multiply(inp, inp)
        abs = tf.abs(mul)
        reduce_max = tf.math.reduce_max(abs, axis=2)
        return reduce_max

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [37, 1, 1, 15, 36] : int8
        trans1 = tf.transpose(inp, perm=[1, 0, 2, 3, 4])
        trans_mul = tf.multiply(trans1, trans1)
        mul = tf.transpose(trans_mul, perm=[1, 0, 2, 3, 4])
        abs = tf.abs(mul)
        reduce_max = tf.math.reduce_max(abs, axis=2)
        argmax = tf.math.argmax(mul, axis=4)
        return reduce_max # removing eitheor trans1 or argmax will not trigger error!

inputs = [
tf.cast(tf.random.uniform(shape=[37, 1, 1, 15, 36], minval=-128, maxval=128, dtype=tf.int32), tf.int8),
]
model1 = Model1()
model2 = Model2()
device = "cpu" # "gpu" also trigger the error
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
import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

densenet = tf.keras.layers.Dense(units=1, dtype=tf.float32, autocast=False)
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        cast = tf.cast(inp, dtype=tf.float32)
        dense = densenet(cast)
        add = tf.add(dense, cast)
        reduce_min = tf.math.reduce_min(add, axis=1)
        return reduce_min,

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        cast = tf.cast(inp, dtype=tf.float32)
        ceil = tf.math.ceil(cast)
        dense = densenet(cast)
        add = tf.add(cast, dense)
        trans1 = tf.transpose(add, perm=[1, 0])
        reduce_min = tf.math.reduce_min(add, axis=1)
        return reduce_min, ceil, trans1,
shape = [45, 29]
inputs = [
tf.complex(tf.random.uniform(shape, dtype=tf.float64), tf.random.uniform(shape, dtype=tf.float64)),
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
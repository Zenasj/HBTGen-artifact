import random
from tensorflow import keras

import tensorflow as tf
import os
import numpy as np

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2, inp3):
        # inp3*(inp2*inp1)*(inp3*(inp2*inp1))
        mul3 = tf.multiply(tf.multiply(inp3, tf.multiply(inp1, inp2)), tf.multiply(inp3, tf.multiply(inp1, inp2)))
        _abs = tf.abs(mul3)
        return mul3, _abs

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2, inp3):
        # inp3 * ((inp1*inp2)*(inp3*(inp1*inp2)))
        mul3 = tf.multiply(inp3, tf.multiply(tf.multiply(inp1, inp2), tf.multiply(inp3, tf.multiply(inp1, inp2))))
        _abs = tf.abs(mul3)
        return mul3, _abs

with tf.device(tf.config.list_logical_devices('GPU')[0].name):
    inputs = [
    tf.random.uniform(shape=[3], minval=-100, maxval=100, dtype=tf.int32),
    tf.random.uniform(shape=[], minval=-100, maxval=100, dtype=tf.int32),
    tf.random.uniform(shape=[3], minval=-100, maxval=100, dtype=tf.int32),
    ]
    model1 = Model1()
    model2 = Model2()
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

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        # Forward pass logic using TensorFlow operations
        _abs = tf.abs(tf.multiply(tf.multiply(inp1, inp2), tf.multiply(inp1, inp2)))
        return _abs

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        # Forward pass logic using TensorFlow operations
        _abs = tf.abs(tf.multiply(inp1, tf.multiply(inp2, tf.multiply(inp1, inp2))))
        return _abs

inputs = [
tf.cast(tf.random.uniform(shape=[11], minval=-128, maxval=128, dtype=tf.int32), tf.int16),
tf.cast(tf.random.uniform(shape=[], minval=-128, maxval=128, dtype=tf.int32), tf.int16),
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
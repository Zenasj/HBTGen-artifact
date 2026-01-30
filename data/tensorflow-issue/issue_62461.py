import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, v4_0):
        v0_0, _ = tf.linalg.eigh(v4_0)
        return v0_0

inputs = [
    tf.complex(tf.random.uniform([1, 48, 1, 1], dtype=tf.float32), 
               tf.random.uniform([1, 48, 1, 1], dtype=tf.float32)),
]

model1 = Model1()
with tf.device('cpu'):
    # Test in eager execution mode
    tf.config.run_functions_eagerly(True)
    out1 = model1(*inputs)
    out2 = model1(*inputs)
    print(f'=========eager_output(version:{tf.__version__})================')
    try:
        for i in range(min(len(out1), len(out2))):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
        print("XLA_eager does not trigger assertion")
    except AssertionError as e:
        print("XLA_eager triggers assertion")
        print(e)

    # Test in compiled mode
    tf.config.run_functions_eagerly(False)
    out1 = model1(*inputs)
    out2 = model1(*inputs)
    print(f'=========compiled_output(version:{tf.__version__})================')
    try:
        for i in range(min(len(out1), len(out2))):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
        print("XLA_compile does not trigger assertion")
    except AssertionError as e:
        print("XLA_compile triggers assertion")
        print(e)
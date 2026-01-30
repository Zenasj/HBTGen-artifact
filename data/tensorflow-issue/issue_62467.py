from tensorflow import keras

class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        v0_0 = tf.abs(inp)
        v2_0 = tf.negative(inp)
        v4_0 = tf.add(v0_0, inp)
        v5_0 = tf.multiply(v2_0, v4_0)
        return v5_0
# Represents: (abs(inp) + inp) * (-inp)

class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def __call__(self, inp):
        v2_0 = tf.negative(inp)
        v3_0 = tf.abs(inp)
        v4_0 = tf.multiply(v3_0, v2_0)
        v5_0 = tf.multiply(inp, v2_0)
        v6_0 = tf.add(v4_0, v5_0)
        return v6_0
# Represents: -inp * abs(inp) + inp * -inp

## After download the pickle file 

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [17, 64, 59, 1, 1] : float32
        v0_0 = tf.abs(inp)
        v2_0 = tf.negative(inp)
        v4_0 = tf.add(v0_0, inp)
        v5_0 = tf.multiply(v2_0, v4_0)
        return v5_0

# (abs(inp) + inp) * (-inp )

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [17, 64, 59, 1, 1] : float32
        v2_0 = tf.negative(inp)
        v3_0 = tf.abs(inp)
        v4_0 = tf.multiply(v3_0, v2_0)
        v5_0 = tf.multiply(inp, v2_0)
        v6_0 = tf.add(v4_0, v5_0)
        return v6_0
# -inp * abs(inp) + inp * -inp 

model1 = Model1()
model2 = Model2()

pickle_file_path = YOUR_PICKLE_FILE_PATH
if not os.path.exists(pickle_file_path) :
    print(f'Pickle file not exist')
else :
    with open(pickle_file_path, 'rb') as f :
        np_arrs1 = pickle.load(f)
    inputs = [tf.convert_to_tensor(arr) for arr in np_arrs1.values()]
    with tf.device('cpu'):
        tf.config.run_functions_eagerly(True)
        out1 = model1(*inputs)
        out2 = model2(*inputs)
        print(f'=========eager_output(version:{tf.__version__})================')
        try :
            for i in range(min(len(out1),len(out2))):
                np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
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
                np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
            print("XLA_complie does not trigger assertion")
        except AssertionError as e:
            print("XLA_complie triggers assertion")
            print(e)
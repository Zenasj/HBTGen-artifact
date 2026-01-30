import math
from tensorflow import keras

import tensorflow as tf
import pickle
import os
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        conc = tf.concat([inp2, inp1], axis=4)
        reduced = tf.math.reduce_prod(conc, axis=4)
        taned = tf.tan(reduced)
        return taned,

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        transposed_inp1 = tf.transpose(inp1, perm=[4, 1, 2, 3, 0])
        transposed_inp2 = tf.transpose(inp2, perm=[4, 1, 2, 3, 0])
        transposed_conc = tf.concat([transposed_inp2, transposed_inp1], axis=0)
        conc = tf.transpose(transposed_conc, perm=[4, 1, 2, 3, 0])
        reduced = tf.math.reduce_prod(conc, axis=4)
        taned = tf.tan(reduced)
        return taned, conc,

model1 = Model1()
model2 = Model2()
device = tf.device(tf.config.list_logical_devices('CPU')[0].name)
pickle_file_path = 'extra_transpose_output_err.pickle' #YOUR_PICKLE_FILE_PATH
if not os.path.exists(pickle_file_path) :
    print(f'Pickle file not exist')
else :
    with open('extra_transpose_output_err.pickle', 'rb') as f :
        nparr = pickle.load(f)
    inputs = [tf.convert_to_tensor(arr) for arr in nparr]
    with device:
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
        xla_out1 = model1(*inputs)
        xla_out2 = model2(*inputs)
        print(f'=========compiled_output(version:{tf.__version__})================')
        try :
            for i in range(min(len(xla_out1),len(xla_out2))):
                np.testing.assert_allclose(xla_out1[i].numpy(), xla_out2[i].numpy(), rtol=0.001, atol=0.001, err_msg=f'at checking {i}th')
            print("XLA_eager does not trigger assertion")
        except AssertionError as e:
            print("XLA_eager triggers assertion")
            print(e)
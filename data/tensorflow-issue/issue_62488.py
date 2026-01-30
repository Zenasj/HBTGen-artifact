import random
from tensorflow import keras

import tensorflow as tf
import pickle
import os
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.p0 = tf.random.uniform(shape=[18, 14, 14, 4], dtype=tf.float32)

    @tf.function(jit_compile=True)
    def __call__(self, inp, inp1):
        astconv = tf.nn.atrous_conv2d(self.p0, inp1, rate=1, padding="VALID")
        _cos = tf.cos(astconv)
        mul = tf.multiply(_cos, astconv)
        return astconv, _cos, mul

model1 = Model1()
device = "gpu"
print(f'=========RUNNING WITH PICKLE FILES===========')
pickle_file_path = 'ast_cos_mul.pickle' #YOUR_PICKLE_FIlE_PATH
if not os.path.exists(pickle_file_path) :
    print(f'Pickle file not exist')
else :
    with open(pickle_file_path, 'rb') as f :
        inputs = pickle.load(f)
    inputs = [tf.convert_to_tensor(arr) for arr in inputs]
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
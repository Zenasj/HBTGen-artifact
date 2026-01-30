import random
from tensorflow import keras

from typing import Dict
import tensorflow as tf
import pickle
import os
import numpy as np

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)
        self.p1 = tf.Variable(tf.random.uniform(shape=[2, 34, 35, 25], dtype=tf.float32)) # [2, 34, 35, 25] float32

        # Layers or other Keras model objects

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [14, 2, 25, 53] : float32
        astconv = tf.nn.atrous_conv2d(self.p1, inp, rate=2, padding="VALID")
        round_ast = tf.round(astconv)
        return astconv, round_ast 

inputs = [
tf.random.uniform(shape=[14, 2, 25, 53], dtype=tf.float32),
]
model1 = Model1()
device = "gpu"
pickle_file_path = "ast_round.pickle" #YOUR_PICKLE_FILE_PATH
if not os.path.exists(pickle_file_path) :
    print(f'Pickle file not exist')
else :
    with open(pickle_file_path, 'rb') as f :
        oracle = pickle.load(f)
    inputs = [tf.convert_to_tensor(arr) for arr in oracle.values()]
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
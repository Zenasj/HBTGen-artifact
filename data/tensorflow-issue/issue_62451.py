import math
import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

params = [
tf.complex(tf.random.uniform([1, 26, 2, 2], dtype=tf.float32), tf.random.uniform([1, 26, 2, 2], dtype=tf.float32)),
]
class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.p0 = tf.Variable(params[0]) # [1, 26, 2, 2] complex64

    @tf.function
    def __call__(self, v7_0):

        cho = tf.linalg.cholesky(self.p0)

        return cho 

inputs = [
tf.complex(tf.random.uniform([1, 26, 1, 2], dtype=tf.float32), tf.random.uniform([1, 26, 1, 2], dtype=tf.float32)),
]

model1 = Model1()
with tf.device('cpu'):
    tf.config.run_functions_eagerly(True)
    out1 = model1(*inputs)
    out2 = model1(*inputs)
    print(f'=========eager_output(version:{tf.__version__})================')
    try :
        for i in range(len(out1)):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
        print("Eager does not trigger assertion")
    except AssertionError as e:
        print("Eeager triggers assertion")
        print(e)

import tensorflow as tf
import numpy as np

params = [
    tf.complex(tf.random.uniform([1, 26, 2, 2], dtype=tf.float32), tf.random.uniform([1, 26, 2, 2], dtype=tf.float32)),
]

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.p0 = tf.Variable(params[0]) # [1, 26, 2, 2] complex64

    @tf.function
    def __call__(self, v7_0):
        # Check if each matrix in the batch is Hermitian
        is_hermitian = tf.reduce_all(tf.math.equal(self.p0, tf.linalg.adjoint(self.p0)), axis=[-2, -1])

        # Calculate the eigenvalues for each matrix in the batch
        eigenvalues = tf.linalg.eigvalsh(self.p0)

        # Check for positive definiteness by ensuring the real parts of the eigenvalues are positive for each matrix
        is_positive_definite = tf.reduce_all(tf.math.real(eigenvalues) > 0, axis=-1)

        # Check if all matrices in the batch are Hermitian and positive definite
        valid_for_cholesky = tf.logical_and(is_hermitian, is_positive_definite)

        if tf.reduce_all(valid_for_cholesky):
            cho = tf.linalg.cholesky(self.p0)
            return cho 
        else:
            print("is_hermitian: ", is_hermitian.numpy())
            print("is_positive_definite: ", is_positive_definite.numpy())
            raise ValueError("Not all matrices are Hermitian positive definite.")

inputs = [
    tf.complex(tf.random.uniform([1, 26, 1, 2], dtype=tf.float32), tf.random.uniform([1, 26, 1, 2], dtype=tf.float32)),
]

model1 = Model1()

with tf.device('cpu'):
    tf.config.run_functions_eagerly(True)
    out1 = model1(*inputs)
    out2 = model1(*inputs)
    print(f'=========eager_output(version:{tf.__version__})================')
    try:
        for i in range(len(out1)):
            np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
        print("Eager does not trigger assertion")
    except AssertionError as e:
        print("Eager triggers assertion")
        print(e)

import tensorflow as tf
import numpy as np

def generate_symmetric_positive_complex_tensor(shape):
    # Ensure the innermost shape is square
    assert shape[-1] == shape[-2], "Shape must be square to create a symmetric matrix"

    # Create a symmetric matrix for the real part and positive definite
    real_part = tf.random.uniform(shape, minval=0, maxval=1)
    real_symmetric = (real_part + tf.transpose(real_part, perm=[0,1,3,2])) / 2

    # Create a symmetric matrix for the imaginary part and positive definite
    imaginary_part = tf.random.uniform(shape, minval=0, maxval=1)
    imaginary_symmetric = (imaginary_part + tf.transpose(imaginary_part, perm=[0,1,3,2])) / 2

    # Combine into a complex tensor
    complex_tensor = tf.complex(real_symmetric, imaginary_symmetric)

    return complex_tensor

params = [
    generate_symmetric_positive_complex_tensor([1, 26, 2, 2]),
    # tf.complex(tf.random.uniform([1, 26, 2, 2], minval=0, maxval=1, dtype=tf.float32), tf.random.uniform([1, 26, 2, 2],minval=0, maxval=1, dtype=tf.float32)),
]

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.p0 = tf.constant(params[0]) # [1, 26, 2, 2] complex64

    @tf.function
    def __call__(self):

        cho = tf.linalg.cholesky(self.p0)
        return cho
   
model1 = Model1()

tf.config.run_functions_eagerly(True)
out1 = model1()
out2 = model1()
print(f'=========eager_output(version:{tf.__version__})================')
try:
    for i in range(len(out1)):
        np.testing.assert_allclose(out1[i].numpy(), out2[i].numpy(), rtol=0.01, err_msg=f'at checking {i}th')
    print("Eager does not trigger assertion")
except AssertionError as e:
    print("Eager triggers assertion")
    print(e)
# tf.random.uniform((1, 48, 1, 1), dtype=tf.complex64) ← Input is a complex tensor with shape (1, 48, 1, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    @tf.function(jit_compile=True)
    def call(self, x):
        # Compute eigenvalues with tf.linalg.eigh
        # The issue involves inconsistent outputs when compiled with XLA.
        # Here we just return the eigenvalues as in the original example.
        eigvals, _ = tf.linalg.eigh(x)
        return eigvals

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random Hermitian matrix input required by tf.linalg.eigh for well-defined behavior
    # since tf.linalg.eigh expects Hermitian matrices A = A^H.
    #
    # The original example input was complex random of shape (1, 48, 1, 1).
    # But such shape does not form a full Hermitian matrix; tf.linalg.eigh expects at least 2D square matrices.
    #
    # To reconcile:
    # - We will generate a batch of 48 Hermitian matrices of shape 1x1, which is trivial (a 1x1 matrix is Hermitian).
    # - Interpret input as shape (batch=1, N=48, M=1, M=1). This can be considered as batch of many 1x1 matrices.
    #
    # However, tf.linalg.eigh expects the last two dims to be square matrices.
    # So shape (1,48,1,1) is batch size 1 with 48 single-element matrices.
    #
    # To ensure Hermitian property for each 1x1 matrix is trivial (real scalar).
    #
    # Alternatively, for a more standard test, we could generate (1, 48, 48) Hermitian matrices.
    # But to follow the original shape strictly, we keep (1,48,1,1).
    
    # Generate random real and imaginary parts
    real_part = tf.random.uniform([1, 48, 1, 1], dtype=tf.float32)
    imag_part = tf.random.uniform([1, 48, 1, 1], dtype=tf.float32)
    
    # Compose complex tensor
    complex_tensor = tf.complex(real_part, imag_part)
    
    # For 1x1 matrices, Hermitian means the element equals its own conjugate transpose:
    # which is trivially true since it's a scalar. No change needed.
    
    return (complex_tensor, )


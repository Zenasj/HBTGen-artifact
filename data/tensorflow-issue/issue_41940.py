import tensorflow as tf

def trisolve(A, b):
    """ Builds a graph. A is a lower-triangular MxM matrix, b is a Mx1 column vector """    
    res = tf.linalg.triangular_solve(A, b, lower=True)    
    return res

M = 2048
predict_fn = tf.function(trisolve,
        input_signature=[tf.TensorSpec(shape=[M,M], dtype=tf.float64, name='A'),
        tf.TensorSpec(shape=[M,1], dtype=tf.float64, name='b')], experimental_compile=False)
    
module_to_save = tf.Module()
module_to_save.predict = predict_fn
tf.saved_model.save(module_to_save, 'saved_model', signatures={'serving_default': module_to_save.predict})

def triangular_solve(L,b):
    """ Solves the equation Lx = b given that L is lower triangular
    
    Parameters:
        - L: a tensor with dimensions [..., M, M]
        - b: a tensor with dimensions [..., M, N]
    """
    # Make all matrices with unitary diagonal
    d = tf.linalg.diag_part(L)
    d = tf.expand_dims(tf.pow(d,-1), axis=-1)
    b = b*d
    L = L*d

    M = tf.shape(b)[-2]
    x = tf.TensorArray(tf.float64, size=M, clear_after_read=False, dynamic_size=False)
    
    # Move last two axis in front    
    b = tf.einsum("...ij->ij...", b)
    L = tf.einsum("...ij->ij...", L)
    
    x = x.unstack(tf.zeros_like(b))                
    L = tf.expand_dims(L, 2)

    def body_fn(i, x):
        coeffsum = tf.reduce_sum(tf.multiply(tf.gather_nd(L, [i]), x.stack()), axis=0)
        x = x.write(i, tf.gather_nd(b, [i]) - coeffsum)
        i = i+1
        return i,x
    
    cond_fn = lambda i,*_: tf.less(i, M)
    _,x = tf.while_loop(cond_fn, body_fn, (tf.constant(0), x))

    return tf.einsum("ij...->...ij", x.stack())

class ModelA():
    def __init__(self, L):
        """ L is a lower triangular MxM matrix """
        self.L = L
        M = self.L.shape[-1]
        self.invL = tf.linalg.triangular_solve(self.L, tf.eye(M, dtype=tf.float64), lower=True)    # Computes the inverse of L
    
    def predict(self, b):        
        """ b is a Mx1 vector """                
        return tf.matmul(self.invL, b)

class ModelA1():
    def __init__(self, L):
        """ L is a lower triangular MxM matrix """
        self.L = L
    
    def predict(self, b):        
        """ b is a Mx1 vector """                
        M = self.L.shape[-1]
        invL = tf.linalg.triangular_solve(self.L, tf.eye(M, dtype=tf.float64), lower=True)    # Computes the inverse of L
        return tf.matmul(invL, b)

class ModelB():
    def __init__(self, L):
        """ L is a lower triangular MxM matrix """
        self.L = L
    
    def predict(self, b):        
        """ b is a Mx1 vector """                
        invL = tf.linalg.inv(self.L)    # Computes the inverse of L
        return tf.matmul(invL, b)

import numpy as np

class ModelC():
    def __init__(self, L):
        """ L is a lower triangular MxM matrix """
        self.L = L
    
    def predict(self, b):        
        """ b is a Mx1 vector """                
        M = self.L.shape[-1]
        eye = tf.constant( np.eye(M) , dtype=tf.float64)   # Use numpy to generate an eye matrix
        invL = tf.linalg.triangular_solve(self.L, eye, lower=True)    # Computes the inverse of L
        return tf.matmul(invL, b)

def my_triangular_solve(A, b):
    S = tf.shape(A)[0]
    ret = tf.zeros(S, dtype=tf.float64)

    for i in tf.range(S):
        acc = tf.reduce_sum(A[i,:] * ret)
        ret = tf.tensor_scatter_nd_update(ret, [[i]], (b[i] - acc) / A[i,i])
    
    return tf.reshape(ret, (1,-1))
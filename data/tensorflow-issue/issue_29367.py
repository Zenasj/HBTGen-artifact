import tensorflow as tf

class A(tf.Module):
    def __init__(self, variable: tf.Tensor):
        self.diag_op = tf.linalg.LinearOperatorLowerTriangular(variable)  # Doesn't work! No links in the gradient between the variable and the rest of the computation, because LinearOperatorLowerTriangular apply all sorts of operations to the input.
# tf.random.uniform((200,), dtype=tf.int32) ← input 'a'; tf.random.uniform((200,), dtype=tf.float32) ← input 's'

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Size of the embedding matrices and vectors
        self.n = 8000

        # Following the last code snippet: 
        #   variables similar to pe, ne, and kp from the issue
        #   pe and ne start as identity matrices (each n x n), trainable tf.Variables
        #   kp starts as zeros vector (length n), trainable tf.Variable
        # The issue examples clip values between -30 and 30 after gradient updates.
        self.pe = tf.Variable(np.eye(self.n), trainable=True, dtype=tf.float32)
        self.ne = tf.Variable(np.eye(self.n), trainable=True, dtype=tf.float32)
        self.kp = tf.Variable(np.zeros(self.n), trainable=True, dtype=tf.float32)

    @tf.function
    def loss_op(self, k: tf.Tensor, a: tf.Tensor, s: tf.Tensor):
        # Computes loss over 200 iterations of a while loop as per provided logic.
        # a: integer tensor indices, s: float tensor signals
        l = tf.constant(0.0)
        
        def loop_fn(i, k, l):
            # Clip predicted probability between 0.01 and 0.99 to avoid log(0)
            p = tf.clip_by_value(k[a[i]], 0.01, 0.99)
            # Binary cross-entropy loss component for sample i
            l = l - (s[i] * tf.math.log(p) + (1 - s[i]) * tf.math.log(1 - p))
            # Update k vector using pe, ne and signals s[i], clipped -30 to 30
            k = tf.clip_by_value(
                k + s[i] * self.pe[a[i]] + (1 - s[i]) * self.ne[a[i]],
                -30.0, 30.0)
            return i + 1, k, l
        
        def loop_cond(i: tf.Tensor, _, __):
            # Run loop while s[i]>=0 and i<199 (200 iterations max)
            return tf.logical_and(tf.greater_equal(s[i], 0), tf.less(i, 199))
        
        _, _, l = tf.while_loop(loop_cond, loop_fn, (0, k, l), back_prop=True)
        return l

    @tf.function
    def regularizer(self, tensor: tf.Tensor):
        # Regularization term sum log(abs(x)+1)
        return tf.reduce_sum(tf.math.log(tf.abs(tensor) + 1))

    @tf.function
    def call(self, inputs):
        # Expect inputs tuple (a, s)
        a, s = inputs
        # Compute loss using current kp vector
        loss = self.loss_op(self.kp, a, s)
        # Regularizers for pe and ne embeddings
        pel = self.regularizer(self.pe)
        nel = self.regularizer(self.ne)
        # Total loss combines these with weighting factor 0.5
        total_loss = loss + 0.5 * (pel + nel)
        return total_loss

    @tf.function
    def train_op(self, a, s, opt):
        # Performs a single training step on inputs (a,s) using optimizer opt
        with tf.GradientTape() as tape:
            loss = self.loss_op(self.kp, a, s)
            pel = self.regularizer(self.pe)
            nel = self.regularizer(self.ne)
            total_loss = loss + 0.5 * (pel + nel)

        train_vars = [self.pe, self.ne, self.kp]
        gradients = tape.gradient(total_loss, train_vars)
        opt.apply_gradients(zip(gradients, train_vars))

        # Clip variables to [-30, 30] after update to keep stability
        self.pe.assign(tf.clip_by_value(self.pe, -30.0, 30.0))
        self.ne.assign(tf.clip_by_value(self.ne, -30.0, 30.0))
        self.kp.assign(tf.clip_by_value(self.kp, -30.0, 30.0))

        # Return original loss (without regularizers) as metric
        return loss


def my_model_function():
    # Instantiate and return MyModel instance
    return MyModel()


def GetInput():
    # Returns valid input tuple (a, s) matching model expectation:
    # a is int32 tensor shape (200,) with indices in [0, 199] (safe indexing)
    # s is float32 tensor shape (200,) with values >=0 (as loop_cond checks s[i]>=0)
    # Use uniform integers for a and ones for s as in the original script
    a = tf.random.uniform(shape=(200,), minval=0, maxval=200, dtype=tf.int32)
    s = tf.ones(shape=(200,), dtype=tf.float32)
    return (a, s)


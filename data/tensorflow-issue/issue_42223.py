# tf.random.uniform((None, 8), dtype=tf.float64) ← The input x has shape (?, 8) based on provided data y shape and usage

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, max_num_iter=100, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.max_num_iter = max_num_iter

        # Define matrix A as a constant tensor
        n = 8
        a_np = np.eye(n, k=0) * 2 - np.eye(n, k=1) - np.eye(n, k=-1)
        a_np[-1, 0] = -1
        a_np[0, -1] = -1
        a_np *= n ** 2
        self.A = tf.constant(a_np, dtype=tf.float64)

        # Alpha step size as per original code = 2e-3
        self.alpha = 2e-3

    def func(self, x):
        # Matrix multiply x by A
        return tf.matmul(x, self.A)

    def loss_func(self, x, y):
        # Loss: mean of norm(func(x)-y)/norm(y) along axis 1
        numerator = tf.norm(self.func(x) - y, axis=1)
        denominator = tf.norm(y, axis=1)
        return tf.reduce_mean(numerator / denominator)

    @tf.function
    def train_one_step(self, x, y, it):
        # Perform one iterative update step depending on iteration index passed explicitly.
        # This fixes the issue of curr_iter not updating when tf.function is applied.

        # Replicate the iterative update x_{k+1} = x_k + alpha * (b - A x_k)
        # Since original call uses curr_iter iterations, here we apply all iterations within tf.function by passing iter count
        for _ in tf.range(it):
            r = y - self.func(x)
            x = x + self.alpha * r
        loss = self.loss_func(x, y)
        return loss, x

    def train(self, x, y):
        # Explicitly run iterative training steps, passing iteration number to train_one_step
        # We accumulate updated x after each step to keep state for the next iteration.
        # This loop is not decorated with tf.function to ensure eager stepwise updates.

        for it in range(1, self.max_num_iter + 1):
            loss, x = self.train_one_step(x, y, it=1)
            print("Step %2d: loss %14.8e" % (it - 1, loss.numpy()))


    def call(self, x, y):
        # Call method applies iterative update max_num_iter times, similar to original
        for _ in range(self.max_num_iter):
            r = y - self.func(x)
            x = x + self.alpha * r
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a zero tensor of shape (batch_size=8, features=8) with dtype float64 as input
    # This matches the example input x shape in the issue
    batch_size = 8
    feature_dim = 8
    return tf.zeros((batch_size, feature_dim), dtype=tf.float64)


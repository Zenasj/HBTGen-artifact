from tensorflow import keras

import tensorflow as tf
import numpy as np


class MyFakeNN(tf.keras.Model):
    def __init__(self, max_num_iter=100, **kwargs):
        super(MyFakeNN, self).__init__(**kwargs)
        self.max_num_iter  = max_num_iter
        self.func = func
        self.curr_iter = 0

    @tf.function
    def train_one_step(self, x, y):
        x_next = self(x, y)
        loss = loss_func(x_next, y)
        return loss

    def train(self, x, y):
        for it in range(self.max_num_iter):
            self.curr_iter = it
            self.trainable = False
            loss = self.train_one_step(x, y)
            print("Step %2d: loss %14.8e" % (it, loss.numpy()))

    def call(self, x, y):
        for _ in range(self.curr_iter):
            r = y - self.func(x)
            x += r * 2e-3
        return x


n = 8
tf.keras.backend.set_floatx('float64')

a = np.eye(n, k=0) * 2 - np.eye(n, k=1) - np.eye(n, k=-1)
a[-1, 0] = -1
a[0, -1] = -1
a *= n ** 2
func = lambda x: tf.matmul(x, a)


def loss_func(x, y):
    return tf.reduce_mean(tf.norm(func(x) - y, axis=1)/tf.norm(y, axis=1))


model = MyFakeNN()
model.trainable = False

y = np.array([[-9.38155831, -28.13152345, 7.78468155, 29.26619534,
               -4.5831364, -25.87319867, 6.18001317, 24.73852678], [-9.83892541, -20.53029054, 20.69956084, 28.01860432,
               -20.97012137, -35.88954846, 10.10948594, 28.40123469], [-21.04290567, -8.05650662, 16.85368166, 1.02738802,
               -22.60513251, -2.13204921, 26.79435652, 9.16116781], [2.04530187, -2.93405974, -2.79706502, -1.04454986,
               -2.07777548, -1.87090609, 2.82953863, 5.84951569], [-4.84959288, 6.48666613, 8.21196668, -4.87053449,
               -9.2887852, 1.73155149, 5.92641139, -3.34768314], [7.66043654, 6.05170581, -3.87066543, 0.99382044,
               10.04477309, 0.69216011, -13.8345442, -7.73768635], [2.54742214, -13.73492348, 0.23540163, 14.91811095,
               -1.34494564, -17.6704306, -1.43787813, 16.48724313], [12.9314438, 3.94013919, -22.58640847, -11.10766707,
               22.104958, 17.59432117, -12.44999333, -10.42679329]])
x = np.zeros(y.shape, dtype='float64')

model.train(x=x, y=y)
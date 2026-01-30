import math

import tensorflow as tf
import numpy as np


class C:
    def __init__(self):
        n = 2000
        self.ae = tf.Variable(np.eye(n), trainable=True, dtype=tf.float32)
        self.aa = tf.Variable(np.eye(n), trainable=True, dtype=tf.float32)
        self.fr = tf.Variable(0.5, trainable=True, dtype=tf.float32)

        self.kp = tf.Variable(np.zeros(n), trainable=False, dtype=tf.float32)

    @tf.function
    def loss_op(self, k: tf.Tensor, a: tf.Tensor, s: tf.Tensor):
        l = tf.constant(0.0)

        def loop_fn(i, k, l):
            p = tf.clip_by_value(k[a[i]], 0.01, 0.99)
            l = l - (s[i] * tf.math.log(p) + (1 - s[i]) * tf.math.log(1 - p))

            at = self.aa[a[i]]

            gk = (self.ae[a[i]] - k) * at
            lk = k * at

            k = tf.clip_by_value(
                k + s[i] * gk - (1 - s[i]) * self.fr * lk, 0.0, 1.0)
            return i + 1, k, l

        def loop_cond(i: tf.Tensor, _, __):
            return tf.logical_and(tf.greater_equal(s[i], 0), tf.less(i, 199))

        _, _, l = tf.while_loop(loop_cond, loop_fn, (0, k, l), back_prop=True)
        return l

    @tf.function
    def regularizer(self, tensor: tf.Tensor):
        return tf.reduce_sum(tf.math.log(tf.abs(tensor) + 1))

    @tf.function
    def train_op(self, a, s, opt):
        with tf.GradientTape() as tape:
            loss = self.loss_op(self.kp, a, s)

            aal = self.regularizer(self.aa)
            al = self.regularizer(self.ae)

            o = loss + 0.5 * (aal + al)

        train_vars = [self.ae, self.aa, self.fr]

        gradient = tape.gradient(o, train_vars)
        opt.apply_gradients(zip(gradient, train_vars))

        return loss

c = C()

o = tf.optimizers.Adam(learning_rate=1e-3)

a = np.arange(200, dtype=np.int32)
s = np.ones(200, dtype=np.float32)

c.train_op(a, s, o)

import tensorflow as tf
import numpy as np

print(tf.__version__)


class C:
    def __init__(self):
        n = 8000
        self.pe = tf.Variable(np.eye(n), trainable=True, dtype=tf.float32)
        self.ne = tf.Variable(np.eye(n), trainable=True, dtype=tf.float32)
        self.kp = tf.Variable(np.zeros(n), trainable=True, dtype=tf.float32)

    @tf.function
    def loss_op(self, k: tf.Tensor, a: tf.Tensor, s: tf.Tensor):
        l = tf.constant(0.0)

        def loop_fn(i, k, l):
            p = tf.clip_by_value(k[a[i]], 0.01, 0.99)
            l = l - (s[i] * tf.math.log(p) + (1 - s[i]) * tf.math.log(1 - p))

            k = tf.clip_by_value(
                k + s[i] * self.pe[a[i]] + (1 - s[i]) * self.ne[a[i]], -30.0, 30.0)
            return i + 1, k, l

        def loop_cond(i: tf.Tensor, _, __):
            return tf.logical_and(tf.greater_equal(s[i], 0), tf.less(i, 199))

        _, _, l = tf.while_loop(loop_cond, loop_fn, (0, k, l), back_prop=True)
        return l

    @tf.function
    def regularizer(self, tensor: tf.Tensor):
        return tf.reduce_sum(tf.math.log(tf.abs(tensor) + 1))

    @tf.function
    def train_op(self, a, s, opt):
        with tf.GradientTape() as tape:
            loss = self.loss_op(self.kp, a, s)

            pel = self.regularizer(self.pe)
            nel = self.regularizer(self.ne)

            o = loss + 0.5 * (pel + nel)

        train_vars = [self.pe, self.ne, self.kp]

        gradient = tape.gradient(o, train_vars)
        opt.apply_gradients(zip(gradient, train_vars))

        self.pe.assign(tf.clip_by_value(self.pe, -30, 30))
        self.ne.assign(tf.clip_by_value(self.ne, -30, 30))
        self.kp.assign(tf.clip_by_value(self.kp, -30, 30))

        return loss


c = C()

o = tf.optimizers.Adam(learning_rate=1e-3)

a = np.arange(200, dtype=np.int32)
s = np.ones(200, dtype=np.float32)

c.train_op(a, s, o)
c.train_op(a, s, o)
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class Outer(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mem = None
        self.loss_mem = None

    def loss(self, x):
        return 2*x-1

    @tf.function
    def gradient(self, x):
        self.mem = tf.TensorArray(tf.float32, size=10)
        self.loss_mem = tf.TensorArray(tf.float32, size=10)
        for i in tf.range(10):
            out_i = self.model.call(x)
            loss_i = self.loss(out_i)
            self.mem = self.mem.write(i, out_i)
            self.loss_mem = self.loss_mem.write(i, loss_i)

        ghat = [tf.zeros_like(i, dtype=tf.float32) for i in self.model.trainable_variables]
        for i in tf.reverse(tf.range(10), [0]):
            with tf.GradientTape() as g:
                g.watch(x)
                out_i = self.model.call(x)
            with tf.GradientTape() as g_loss:
                g_loss.watch(out_i)
                loss_i = self.loss(out_i)
            output_grad = g_loss.gradient(loss_i, out_i)
            ghat_update = g.gradient(out_i, self.model.trainable_variables, output_gradients=output_grad)
            for j in range(len(ghat)):
                ghat[j] = ghat[j] + ghat_update[j]
        return ghat

class Inner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.out(x[0])


x = [tf.zeros((32, 4, 16), tf.float32), tf.zeros((64, 10), tf.float32)]
mytest = Outer(Inner())
mytest.gradient(x)

import tensorflow as tf


class Outer(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mem = None
        self.loss_mem = None

    def loss(self, x):
        return 2*x-1

    def test_maybe(self, x):
        self.mem = tf.TensorArray(tf.float32, size=10)
        self.loss_mem = tf.TensorArray(tf.float32, size=10)
        for i in tf.range(10):
            out_i = self.model.call(x)
            loss_i = self.loss(out_i)
            self.mem = self.mem.write(i, out_i)
            self.loss_mem = self.loss_mem.write(i, loss_i)

    @tf.function
    def gradient(self, x):
        # self.test_maybe(x)

        ghat = [tf.zeros_like(i, dtype=tf.float32) for i in self.model.trainable_variables]
        for i in tf.reverse(tf.range(10), [0]):
            with tf.GradientTape() as g:
                g.watch(x)
                out_i = self.model.call(x)
            with tf.GradientTape() as g_loss:
                g_loss.watch(out_i)
                loss_i = self.loss(out_i)
            output_grad = g_loss.gradient(loss_i, out_i)
            ghat_update = g.gradient(out_i, self.model.trainable_variables, output_gradients=output_grad)
            for j in range(len(ghat)):
                ghat[j] = ghat[j] + ghat_update[j]
        return ghat

class Inner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.out = tf.keras.layers.Dense(10)

    @tf.function
    def call(self, x):
        return self.out(x[0])


x = [tf.zeros((32, 4, 16), tf.float32), tf.zeros((64, 10), tf.float32)]
mytest = Outer(Inner())
mytest.gradient(x)

import tensorflow as tf


class Outer(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mem = None
        self.loss_mem = None

    def loss(self, x):
        return 2*x-1

    @tf.function
    def gradient(self, x):
        self.mem = tf.TensorArray(tf.float32, size=10)
        self.loss_mem = tf.TensorArray(tf.float32, size=10)
        for i in tf.range(10):
            out_i = self.model.call(x)
            loss_i = self.loss(out_i)
            self.mem = self.mem.write(i, out_i)
            self.loss_mem = self.loss_mem.write(i, loss_i)

        ghat_old = [tf.zeros_like(i, dtype=tf.float32) for i in self.model.trainable_variables]
        for i in tf.reverse(tf.range(10), [0]):
            ghat = [j for j in ghat_old]
            with tf.GradientTape() as g:
                g.watch(x)
                out_i = self.model.call(x)
            with tf.GradientTape() as g_loss:
                g_loss.watch(out_i)
                loss_i = self.loss(out_i)
            output_grad = g_loss.gradient(loss_i, out_i)
            ghat_update = g.gradient(out_i, self.model.trainable_variables, output_gradients=output_grad)
            for j in range(len(ghat)):
                ghat[j] = ghat[j] + ghat_update[j]
            ghat_old = ghat
        return ghat

class Inner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.out(x[0])


x = [tf.zeros((32, 4, 16), tf.float32), tf.zeros((64, 10), tf.float32)]
mytest = Outer(Inner())
mytest.gradient(x)
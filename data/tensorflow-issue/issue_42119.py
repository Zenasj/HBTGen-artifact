import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

t = tf.Variable(0)

@tf.function()
def foo():
    for n in tf.range(tf.constant(10)):
        t.assign_add(n)
    return t

nt = foo()
nt #  <tf.Tensor: shape=(), dtype=int32, numpy=45>

class mymodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.t = tf.Variable(0)
    def train_step(self, data):
        for n in tf.range(tf.constant(10)):
            t.assign_add(n)
        return {"loss": t}

mm = mymodel()
mm.compile()
mm.fit(np.random.random((5)), steps_per_epoch=1) # this doesn't work see trace below

class mymodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.t = tf.Variable(0)
    @tf.function()
    def train_step(self, data):
        for n in tf.range(tf.constant(10)):
            self.t.assign_add(n)
        return {"loss": self.t}

class mymodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.t = tf.Variable(0)
    @tf.function()
    def train_step(self, data):
        for n in tf.range(tf.constant(10)):
            self.t.assign_add(n)
        return {"loss": self.t}

class TestFunctions:

    def __init__(self, ndims, alpha, **kwargs):
        self.ndims = ndims
        self.alpha = alpha
        self.variables = kwargs
        self.calls = 0

    def loop1(self, x):
        self.calls += 1 
        my_new_temp = tf.cast(wrapper.integrand_1d_python(x), dtype=tf.float64) 

        return my_new_temp
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class opt:
    def __init__(self):
        self.x = None
    @tf.function
    def __call__(self, data):
        if self.x is None:
            self.x = tf.Variable(0., dtype=tf.float64)
            self.opt = tf.keras.optimizers.SGD(1.)
        x = self.x
        opt = self.opt
        for _ in tf.range(10):
            with tf.GradientTape() as tape:
                obj = tf.reduce_mean((data - x) ** 2)
            g = tape.gradient(obj, x)
            opt.apply_gradients([(g, x)])
        return x

data = np.random.normal(size=10000)
K = 10
replicates = np.random.choice(data, size=(K, 10000), replace=True)
opts = [opt() for _ in range(len(replicates))]

@tf.function
def g(replicates):
    return [opts[i](replicates[i]) for i in range(len(opts))]

g(replicates)
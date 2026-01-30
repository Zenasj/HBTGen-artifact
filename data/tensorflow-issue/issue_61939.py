from tensorflow import keras

import tensorflow as tf

"""
Without Autocluster
"""
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.v3_weight = tf.Variable(123.45)

    def call(self, x1):
        x4 = (self.v3_weight * (- 0.0))
        return x4
m = Model()
x = tf.constant(702.89)
print(m(x)) # tf.Tensor(-0.0, shape=(), dtype=float32)

"""
With Autocluster
"""
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.v3_weight = tf.Variable(123.45)
    
    @tf.function
    def call(self, x1):
        x4 = (self.v3_weight * (- 0.0))
        return x4
m = Model()
x = tf.constant(702.89)
print(m(x)) # tf.Tensor(0.0, shape=(), dtype=float32)

py
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.v3_weight = tf.Variable(123.45)

    def call(self, x1):
        x4 = (self.v3_weight * (- 0.0))
        x4 = 1 / x4
        return x4
m = Model()
x = tf.constant(702.89)
print(m(x)) # tf.Tensor(-inf, shape=(), dtype=float32)

py
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.v3_weight = tf.Variable(123.45)

    @tf.function
    def call(self, x1):
        x4 = (self.v3_weight * (- 0.0))
        x4 = 1 / x4
        return x4
m = Model()
x = tf.constant(702.89)
print(m(x)) # tf.Tensor(inf, shape=(), dtype=float32)
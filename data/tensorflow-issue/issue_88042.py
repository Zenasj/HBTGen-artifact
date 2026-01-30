from tensorflow import keras

import os
import tensorflow
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ComplexModel(tf.keras.Model):

    def __init__(self):
        super(ComplexModel, self).__init__()

    def call(self, x):
        x_sparse = tf.sparse.from_dense(x)
        x = tf.sparse.minimum(x_sparse, tf.sparse.from_dense(tf.ones_like(x)))
        return x


model = ComplexModel()



x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

inputs = [x]

model(*inputs)
print("succeed on eager")


class ComplexModel(tf.keras.Model):

    def __init__(self):
        super(ComplexModel, self).__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        x_sparse = tf.sparse.from_dense(x)
        x = tf.sparse.minimum(x_sparse, tf.sparse.from_dense(tf.ones_like(x)))
        return x


model = ComplexModel()
model(*inputs)
print("succeed on XLA")
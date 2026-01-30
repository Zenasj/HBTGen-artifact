import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

#Without this line the script works in v1.14.
#comment out this line for v2.0-rc0
tf.enable_eager_execution()


def matmul_dense_sparse(a, b):
    ta = tf.transpose(a)
    tb = tf.sparse.transpose(b)
    return tf.transpose(tf.sparse.sparse_dense_matmul(tb, ta))


class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, indices, shape):
        super().__init__()
        self.indices = indices
        self.shape = shape

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                        shape=(self.indices.shape[0],),
                        trainable=True)
        self.sparse_mat = tf.sparse.reorder(tf.sparse.SparseTensor(self.indices, self.w, self.shape))
        super().build(input_shape)

    def call(self, inputs):
        return matmul_dense_sparse(inputs, self.sparse_mat)


class SparseModel(tf.keras.Model):
    def __init__(self, indices, shape):
        super().__init__()
        self.l1 = SparseLayer(indices, shape)

    def call(self, inputs):
        return self.l1(inputs)

indices = np.array([[1, 2], [30, 1], [30, 3], [45, 2], [56, 2], [32, 4]])

ex_x = np.random.rand(20, 100)
ex_y = np.random.rand(20, 5)

model = SparseModel(indices, (100, 5))

#compile for v1.14
model.compile(tf.keras.optimizers.SGD(), tf.losses.mean_squared_error)
#compile for v2.0-rc0
#model.compile(tf.keras.optimizers.SGD(), tf.losses.MeanSquaredError())

model.fit(ex_x, ex_y)
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class SqueezedSparseConversion(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.SparseTensor([(0, 1)], [0.1], (3, 3))

class GraphConvolution(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[0]

x_t = tf.keras.Input(0)
sp = SqueezedSparseConversion()(x_t)
out = GraphConvolution()([x_t, sp])

m = tf.keras.Model([x_t], out)
m.summary()
m.save("")

import tensorflow as tf

class GraphConvolution(tf.keras.layers.Layer):
    def call(self, inputs):
        sp = tf.SparseTensor([(0, 1)], [0.1], (3, 3))
        return inputs[0]

x_t = tf.keras.Input(0)
out = GraphConvolution()([x_t])

m = tf.keras.Model([x_t], out)
m.summary()
m.save("")
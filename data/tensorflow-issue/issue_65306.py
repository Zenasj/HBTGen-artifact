from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print(tf.__version__)

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[int(input_shape[-1]),
                                         self.num_outputs], )

    def call(self, inputs):
        kernel = tf.cast(self.kernel, tf.complex64)
        return tf.matmul(inputs, kernel)

layer = MyDenseLayer(10)
layer(tf.zeros([10, 5], dtype=tf.complex64))

import tensorflow as tf
print(tf.__version__)

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[int(input_shape[-1]),
                                         self.num_outputs], )

    def call(self, inputs):
        kernel = tf.cast(self.kernel, tf.complex64)
        return tf.matmul(inputs, kernel)

layer = MyDenseLayer(10)
layer(tf.zeros([10, 5], dtype=tf.complex64))
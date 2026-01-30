from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
class CustomLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1))
    def call(self, x):
        return tf.matmul(x, self.w)
inp = tf.keras.Input((5,))
m = tf.keras.Model(inputs=inp, outputs=CustomLayer()(inp))
m.save("/tmp/savedmodel")
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class A(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(A, self).__init__()
        self.layer = layer

    def call(self, inputs):
        return self.layer(inputs)


class B(tf.keras.layers.Layer):
    def __init__(self):
        super(B, self).__init__()
        self.obj = tf.keras.layers.Dense(13, kernel_regularizer=tf.keras.regularizers.l1(5))
        self.layerB = A(self.obj)

    def call(self, inputs):
        return self.layerB(inputs)

model = B()

output = model(tf.ones([5, 10]))
print(len(model.losses))
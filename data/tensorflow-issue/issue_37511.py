import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self._model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet')

    def build(self, input_shape=None):
        for layer in self._model.layers:
            if type(layer) == tf.keras.layers.Conv2D:
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)

    def call(self, x):
        return self._model(x)


if __name__ == "__main__":
    m = Model()
    x = tf.random.uniform(shape=(1, 512, 512, 3))
    m.build()
    m.call(x)
    print(m.losses)
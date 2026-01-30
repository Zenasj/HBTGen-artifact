import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.d = tf.keras.layers.Dense(2)

    @tf.function
    def call(self, x, training=True, mask=None):
        return self.d(x)


model = Model()
model(tf.random.normal((2, 3)))
# next line raises errors
model.save("save/model", save_format="tf")

model = Model()
# next line !
model(tf.random.normal((2, 3)))
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class inner_block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="inner")
        self.layer = tf.keras.layers.Dense(2, name="inner_dense") # kernel of this will save as "variables/0/.ATTRIBUTES/VARIABLE_VALUE"
        self.var = tf.Variable([1.0, 2.0], name="inner_var") # this will save as expected "inner_m/var/.ATTRIBUTES/VARIABLE_VALUE"

    def call(self, x):
        return self.layer(x) + self.var

class model(tf.keras.Model):
    def __init__(self):
        super().__init__(name="outer")
        self.inner_m = inner_block()
        self.dense = tf.keras.layers.Dense(2, name="outer_dense") # kernel of this will save as expected "dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"

    def call(self, x):
        x = self.inner_m(x)
        x = self.dense(x)
        return x

m = model()
m.build((None, 3))
m(tf.random.normal((1, 3)))

m.save("model")  # or tf.saved_model.save(m, "model")
tf.train.list_variables(
    "model/variables/variables"
)  # this lists 'variables/0/...' 'variables/1/...' along with the expected names
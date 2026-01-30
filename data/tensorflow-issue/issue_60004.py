import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from keras import layers
import numpy as np


class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__()
        self.b = self.add_weight(shape=(1, 5), trainable=True)
        self.a = tf.Variable([1.0], trainable=False)
        self.a.trainable = True

    def call(self, inputs, training=None, mask=None):
        return self.a * self.b * inputs


input = np.random.randn(10, 5)
target = np.random.randn(10, 1)

model = model()
model.b.trainable = True
model.a.trainable = True
model.compile("adam", loss="mse")

model.fit(input, target, validation_data=(input, target), epochs=3)

model.a.trainable = True

import tensorflow as tf
from keras import layers
import numpy as np


class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__()
        self.b = self.add_weight(shape=(1, 5), trainable=True)
        self.a = tf.Variable([1.0], trainable=True)

    def call(self, inputs, training=None, mask=None):
        return self.a * self.b * inputs


input = np.random.randn(10, 5)
target = np.random.randn(10, 1)

model = model()
model.compile("adam", loss="mse")

model.fit(input, target, validation_data=(input, target), epochs=3)

class VarablesLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name, trainable=True, activation=None, **kwargs):
        super(VarablesLayer, self).__init__(name=name, trainable=trainable)
        self.var = self.add_weight(name=f"{name}/var", shape=shape, **kwargs)
        self.activation = activation


    def call(self, inputs, *args, **kwargs):
        if self.activation:
            return self.activation(self.var)
        return self.var

class a:

  def __init__(self,trainable):
    self.trainable = trainable

  @property
  def trainable(self):
    return self._trainable

c = a(False)
c.trainable = True

class a:

  def __init__(self,trainable):
    self.trainable = trainable

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    self._trainable = value


c = a(False)
c.trainable = True
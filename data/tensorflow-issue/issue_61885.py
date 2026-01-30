from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
x1 = tf.constant([[6., 7.]], shape=[1, 2])

##### With @tf.function #####
class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = tf.keras.layers.Dense(2, name='fc', kernel_initializer='ones', bias_initializer='ones')
    self.fc2 = tf.keras.layers.Dense(2, name='fc', kernel_initializer='ones', bias_initializer='ones')

  @tf.function
  def call(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = tf.constant([6.]) + x
    return tf.constant([7.]) + x

m = Model()
y = m(x1)
print(y.numpy())

##### Without @tf.function #####
class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = tf.keras.layers.Dense(2, name='fc', kernel_initializer='ones', bias_initializer='ones')
    self.fc2 = tf.keras.layers.Dense(2, name='fc', kernel_initializer='ones', bias_initializer='ones')

  def call(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = tf.constant([6.]) + x
    return tf.constant([7.]) + x

m = Model()
expected_value = m(x1)
print(expected_value.numpy())
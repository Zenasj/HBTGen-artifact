import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):

  def __init__(self):
    pdb.set_trace()
    super(MyModel, self).__init__()
    pdb.set_trace()

    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    pdb.set_trace()
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    pdb.set_trace()

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)
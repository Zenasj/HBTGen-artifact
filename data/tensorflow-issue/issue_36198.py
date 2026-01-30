from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense = tf.keras.layers.Dense(16)
    self.concat = tf.keras.layers.Concatenate(axis=1)

  def call(self, inputs):
    return self.dense(inputs)

model = MyModel()
model.build((16, 16))
model.summary()
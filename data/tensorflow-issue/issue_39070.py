from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.lstm = tf.keras.layers.LSTM(10)

  def call(self, inputs, initial_state):
    output = self.lstm(inputs, initial_state=initial_state)
    return output

model = Model()
model(tf.zeros((8, 2, 5)), initial_state=[tf.zeros((8, 10)), tf.zeros((8, 10))])
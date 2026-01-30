import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class TestModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.rnn = tf.keras.layers.LSTM(32, return_sequences=True)
  
  def call(self, x, mask=None):
    return self.rnn(x, mask=mask)[:, -1, :]

batch_size = 4
seq_len = 7

x = tf.random.normal([batch_size, seq_len, 32])
model = TestModel()
model(x, mask = tf.sequence_mask([seq_len - 1] * batch_size, maxlen=seq_len))
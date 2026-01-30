import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def gru_cell(enc_units):
  return tf.keras.layers.GRUCell(enc_units, recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.bid_gru = tf.keras.layers.Bidirectional(gru_cell(self.enc_units))

   

  def call(self, x, hidden):
    x = self.embedding(x)
    concatenated, forward, backward = self.bid_gru(x, initial_state=[hidden, hidden])
    return concatenated, forward, backward

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
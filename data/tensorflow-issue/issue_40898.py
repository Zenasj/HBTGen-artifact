import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tfLite

Tx = 8
def Partial_model():
    outputs = []
    X = tf.keras.layers.Input(shape=(Tx,))
    partial = tf.keras.layers.Input(shape=(Tx,))
    enc_hidden = tf.keras.layers.Input(shape=(units,))
    dec_input = tf.keras.layers.Input(shape=(1,))
    
    d_i = dec_input
    e_h = enc_hidden
    X_i = X
    
    enc_output, e_h = encoder(X, enc_hidden)
    
    
    dec_hidden = enc_hidden
    print(dec_input.shape, 'inp', dec_hidden.shape, 'dec_hidd')
    for t in range(1, Tx):
        print(t, 'tt')
      # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(d_i, dec_hidden, enc_output)
#         outputs.append(predictions)
        print(predictions.shape, 'pred')
        d_i = tf.reshape(partial[:, t], (-1, 1))
        print(dec_input.shape, 'dec_input')
    
    predictions, dec_hidden, _ = decoder(d_i, dec_hidden, enc_output)
    d_i = tf.squeeze(d_i)
    
    outputs.append(tf.math.top_k(predictions, 5))
    
    return tf.keras.Model(inputs = [X, enc_hidden, dec_input, partial], outputs = [outputs[0][0], outputs[0][1]])




class Encoder():
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def __call__(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    print(output.shape, hidden.shape, "out", "hid")
    return output, state


  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))



class BahdanauAttention():
  def __init__(self, units):
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def __call__(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    print(query.shape, 'shape')
    query_with_time_axis = tf.expand_dims(query, 1)
    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    print("2")
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    print("3")

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    
    return context_vector, attention_weights


class Decoder():
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def __call__(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)
    
    print(context_vector.shape, 'c_v', attention_weights.shape, "attention_w")

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    print(x.shape, 'xshape', context_vector.shape, 'context')
    expanded_dims = tf.expand_dims(context_vector, 1)
    x = tf.concat([expanded_dims, x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
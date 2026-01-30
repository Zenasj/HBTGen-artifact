from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
# from crf import CRF

class CrfModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CrfModel, self).__init__(*args, **kwargs)


class BiRNNCrf(CrfModel):
    def __init__(self, vocab_size, embedding_dim, cell_creator, num_tags, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.forward_cell = cell_creator()
        self.forward_rnn_layer = tf.keras.layers.RNN(self.forward_cell, return_sequences=True, return_state=False)
        self.backward_cell = cell_creator()
        self.backward_rnn_layer = tf.keras.layers.RNN(self.backward_cell, return_sequences=True, return_state=False, go_backwards=True)
        self.bi_rnn_layer = tf.keras.layers.Bidirectional(self.forward_rnn_layer, backward_layer=self.backward_rnn_layer)
        self.fc = tf.keras.layers.Dense(num_tags)
        # self.crf = CRF(num_tags)

    def lookup(self, inputs):
        out = self.embedding(inputs)
        return out
    
    def birnn(self, inputs, sequence_mask):
        out = self.bi_rnn_layer(inputs, mask=sequence_mask)
        return out

    # @tf.function(input_signature=[tf.TensorSpec([None, None, None], dtype=tf.float32, name="input_ids"), tf.TensorSpec([None], dtype=tf.int32, name="sequence_length")])
    def call(self, inputs, sequence_length):
        out = self.lookup(inputs)
        mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])
        out = self.birnn(out, mask)
        out = self.fc(out)
        # out = self.crf(out, sequence_length)
        return out

def cell_creator():
    return tf.keras.layers.LSTMCell(300)

"""
def build(self, input_shape, *args, **kwargs):
    super(self, tf.keras.Model).build(input_shape, *args, **kwargs)
    self.crf.build(input_shape)
"""

if __name__ == "__main__":
    nn = BiRNNCrf(1000, 300, cell_creator, 3)
    assert isinstance(nn, tf.keras.Model)
    # nn.build(input_shape=[tf.TensorShape([None,None,3]), tf.TensorShape([None])]) # Not allowed by tensorflow
    nn(inputs=tf.constant([[1, 0, 0], [1, 1, 1]]), sequence_length=tf.constant([1,3]))
    print(nn.forward_rnn_layer.built) # False
    print(nn.backward_rnn_layer.built) # True

    nn.summary() # Fails

def __init__(self, vocab_size, embedding_dim, cell_creator, num_tags, *args, **kwargs):
    super().__init__(self, *args, **kwargs)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    forward_rnn_layer = tf.keras.layers.RNN(cell_creator(), return_sequences=True, return_state=False)
    backward_rnn_layer = tf.keras.layers.RNN(cell_creator(), return_sequences=True, return_state=False,
                                                  go_backwards=True)
    self.bi_rnn_layer = tf.keras.layers.Bidirectional(forward_rnn_layer, backward_layer=backward_rnn_layer)
    self.fc = tf.keras.layers.Dense(num_tags)
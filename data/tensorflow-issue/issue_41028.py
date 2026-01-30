import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

class TranNetwork(tf.keras.Model):
    """Transcription Network
    """
    def __init__(self, num_lstm_layers, lstm_cell_size, dropout=0.0):
        super(TranNetwork, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.lstm_cell_size = lstm_cell_size
        self.dropout = dropout
        self.trans_layers = []
        self.pooling = tf.keras.layers.MaxPool1D(pool_size=3, padding="same")

        # HERE IS THE ISSUE, USING RECURRENT ACTIVATION AS tf.keras.activations.sigmoid instead of 'sigmoid'
        # causes model to not converge at all
        recurrent_activation = 'sigmoid'
        #recurrent_activation = tf.keras.activations.sigmoid
        for l in range(num_lstm_layers):
          self.trans_layers.append(tf.keras.layers.LSTM(
                                     self.lstm_cell_size,
                                     return_sequences=True,
                                     dropout=self.dropout,
                                     recurrent_activation=recurrent_activation))

    def call(self, x, x_len, training=True):
        seq_len = x_len
        output = tf.clip_by_value(x, -3.0, 3.0)
        for l in range(self.num_lstm_layers):
            mask = tf.sequence_mask(seq_len)
            output = self.trans_layers[l](output, mask=mask, training=training)
            if l == 0:
              seq_len = tf.cast(tf.math.ceil(tf.divide(seq_len, 3)), dtype=tf.int32)
              output = self.pooling(output, training=training)

        mask = tf.sequence_mask(seq_len)
        output = self.trans_layers[-1](output, mask=mask, training=training)

        return output, seq_len

class RNNTModel(object):
    """ RNNT Model class for training
    """
    def __init__(self, num_lstm_layers, lstm_cell_size, vocab_size, dropout=0.0):
        self.vocab_size = vocab_size
        self.trans = TranNetwork(num_lstm_layers, lstm_cell_size, dropout=dropout)
        self.ctc_layer = tf.keras.layers.Dense(1+self.vocab_size, name='ctc')

    def forward(self, x, y, x_len, y_len, training=True):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, None, 40], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.bool)])
        def _forward(x, y, x_len, y_len, training=True):
            trans_output, output_len = self.trans(x, x_len, training=training)
            ctc_out = self.ctc_layer(trans_output, training=training)
            output_d = {'ctc_out': ctc_out, 'output_len': output_len}
            return output_d
        return _forward(x, y, x_len, y_len, training=training)

    def loss(self, y, x_len, y_len, ctc_out):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None, 1+self.vocab_size], dtype=tf.float32)])
        def _loss(y, x_len, y_len, ctc_out):
            ctc_loss = tf.nn.ctc_loss(y, ctc_out, y_len, x_len,
                                      logits_time_major=False,
                                      blank_index=self.vocab_size)
            # to ignore invalid ctc loss case
            mask = tf.dtypes.cast(
              tf.math.greater_equal(x_len, y_len), dtype=tf.float32)
            ctc_loss = tf.multiply(ctc_loss, mask)
            ctc_loss = tf.reduce_sum(ctc_loss)
            return ctc_loss
        return _loss(y, x_len, y_len, ctc_out)

    @property
    def trainable_variables(self):
        return self.trans.trainable_variables \
               + self.ctc_layer.trainable_variables

model = RNNTModel(2, 32, 29, dropout=0.0)
optimizer = tf.keras.optimizers.Adam(0.00025)

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, 40], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    tf.TensorSpec(shape=[None, ], dtype=tf.int32),
    tf.TensorSpec(shape=[None, ], dtype=tf.int32)])
def train_step(x, y, x_len, y_len):
    with tf.GradientTape() as tape:
        output_d = model.forward(x, y, x_len, y_len, training=True)
        ctc_loss = model.loss(y, output_d['output_len'],
                              y_len, output_d['ctc_out'])
    variables = model.trainable_variables
    gradients = tape.gradient(ctc_loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, variables))
    loss_norm_factor = 1.0 / tf.cast(tf.reduce_sum(y_len), dtype=tf.float32)
    return ctc_loss * loss_norm_factor
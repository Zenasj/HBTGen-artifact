# tf.random.uniform((100, 1403), dtype=tf.int32)

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTMCell, RNN
from tensorflow.keras.models import Model

class LSTMCellwithStates(LSTMCell):
    def call(self, inputs, states, training=None):
        # The inputs to this cell contain concatenated data, but the actual inputs are only
        # the first `units` columns (decoupling [h,c]).
        real_inputs = inputs[:, :self.units]
        outputs, [h, c] = super().call(real_inputs, states, training=training)
        # Concatenate h and c to return combined output for the RNN layer
        return tf.concat([h, c], axis=1), [h, c]

class MyModel(tf.keras.Model):
    def __init__(self, batch_size=100, input_length=1403, embedding_vocab_size=500, embedding_dim=100, lstm_units=200):
        super().__init__()
        self.batch_size = batch_size
        self.input_length = input_length
        self.embedding_vocab_size = embedding_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # Input and embedding layers
        self.input_layer = Input(batch_shape=(batch_size, input_length), name='input')
        self.embedding = Embedding(
            input_dim=self.embedding_vocab_size, output_dim=self.embedding_dim,
            input_length=input_length, trainable=False, name='embedding')
        
        # Custom LSTM cell wrapped in RNN layer
        self.rnn = RNN(LSTMCellwithStates(self.lstm_units), 
                       return_sequences=True, return_state=False, name='LSTM')
        
        # Initial states as Variables so gradients can be computed with respect to them
        self.h0 = tf.Variable(tf.random.uniform((batch_size, lstm_units)), trainable=True)
        self.c0 = tf.Variable(tf.random.uniform((batch_size, lstm_units)), trainable=True)

    def call(self, inputs, training=None):
        # Compute embeddings
        emb_out = self.embedding(inputs)
        # Pass embeddings through RNN with initial states h0 and c0
        # The output's shape is (batch_size, timesteps, 2*lstm_units) due to concatenation of h and c
        rnn_allstates = self.rnn(emb_out, initial_state=[self.h0, self.c0], training=training)
        # Return concatenated [h, c] states per timestep
        return rnn_allstates

    @tf.function
    def compute_dct_dc0(self, ct):
        # Compute gradient of cell state ct w.r.t initial cell state c0 using tf.GradientTape
        # We use GradientTape since tf.gradients (TF1.x) returns None in TF2 for Variables by default
        with tf.GradientTape() as tape:
            tape.watch(self.c0)
            # To get a scalar output for gradient, reduce mean over batch and units
            output_scalar = tf.reduce_mean(ct)
        grad = tape.gradient(output_scalar, self.c0)
        return grad

def my_model_function():
    # Instantiate and return the MyModel instance initialized with default parameters
    return MyModel()

def GetInput():
    # Return input tensor shape (batch_size, input_length) with int32 values in vocab range [0, 500)
    # Matches MyModel embedding input.
    batch_size = 100
    input_length = 1403
    vocab_size = 500
    return tf.random.uniform(shape=(batch_size, input_length), maxval=vocab_size, dtype=tf.int32)


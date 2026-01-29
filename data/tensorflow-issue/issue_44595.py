# tf.random.uniform((batch_sz, sequence_length), dtype=tf.int32) — input is token IDs (integer sequences)

import tensorflow as tf

def gru_cell(enc_units):
    # Returns a GRUCell with 'glorot_uniform' initializer for recurrent kernel
    return tf.keras.layers.GRUCell(enc_units, recurrent_initializer='glorot_uniform')

class MyModel(tf.keras.Model):
    """
    Encoder with a Bidirectional GRU layer.
    
    Inputs:
        x: integer tensor of shape (batch_sz, sequence_length) representing token IDs.
        hidden: initial hidden state tensor of shape (batch_sz, enc_units)
        
    Outputs:
        concatenated: the concatenated output of the bidirectional GRU, shape (batch_sz, sequence_length, enc_units * 2)
        forward_state: final forward GRU hidden state, shape (batch_sz, enc_units)
        backward_state: final backward GRU hidden state, shape (batch_sz, enc_units)
    
    Notes:
    - The model uses an Embedding layer to convert token IDs to embeddings.
    - The Bidirectional wrapper wraps the GRUCell inside a RNN layer.
    - Initial states for forward and backward GRU are both set to the same 'hidden' input.
    """

    def __init__(self, vocab_size=10000, embedding_dim=256, enc_units=1024, batch_sz=64):
        super(MyModel, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Wrap the gru_cell in a RNN layer, then bidirectional wrapper:
        # This corrects the original mistake where Bidirectional was wrapping GRUCell directly.
        self.bid_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(gru_cell(self.enc_units), return_sequences=True, return_state=True)
        )

    def call(self, x, hidden):
        # x: (batch_sz, seq_len) integer token IDs
        x = self.embedding(x)  # (batch_sz, seq_len, embedding_dim)
        # For Bidirectional RNN initial_state, must provide forward and backward initial states separately:
        # Both forward and backward GRUs use the same 'hidden' as initial state here.
        # initial_state must be a list: [forward_state, backward_state]
        concatenated, forward_state, backward_state = self.bid_gru(
            x, initial_state=[hidden, hidden]
        )
        return concatenated, forward_state, backward_state

    def initialize_hidden_state(self):
        # Returns zero tensor for initial hidden state: shape (batch_sz, enc_units)
        return tf.zeros((self.batch_sz, self.enc_units))

def my_model_function():
    # Return an instance of MyModel with default parameters
    # Defaults chosen as common example values
    return MyModel()

def GetInput():
    # Generate example input tensor of integer token IDs with shape (batch_sz, sequence_length)
    batch_sz = 64
    sequence_length = 10
    vocab_size = 10000
    # Integer IDs between 0 and vocab_size-1
    x = tf.random.uniform(
        (batch_sz, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    # Instantiate model to get enc_units and batch_sz for initial hidden
    model = my_model_function()
    hidden = model.initialize_hidden_state()
    # Return tuple of input tensor and initial hidden state to match call signature
    return (x, hidden)


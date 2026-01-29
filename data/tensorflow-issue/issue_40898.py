# tf.random.uniform((1, 8), dtype=tf.int32) ← Input shape is (batch=1, Tx=8) integer token ids for encoder input

import tensorflow as tf

Tx = 8  # input sequence length
units = 1024  # assumed hidden unit size (from snippet variables)
vocab_size = 10000  # Assumed vocabulary size (not specified, typical medium vocab size)
embedding_dim = 256  # typical embedding dimension, inferred plausible


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_len, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # x shape: (batch_size, 1) token ids
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)
        expanded_context = tf.expand_dims(context_vector, 1)
        # Concatenate on last axis: (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([expanded_context, x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)  # (batch_size, vocab_size)
        return x, state, attention_weights


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.batch_size = 1  # fixed batch size for input/output consistency
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.units, self.batch_size)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.units, self.batch_size)

    def call(self, inputs):
        """
        inputs: tuple/list of 4 tensors
            X: (batch_size=1, Tx=8) int32 encoder input sequence (token IDs)
            enc_hidden: (batch_size=1, units=1024) float32 initial encoder hidden state
            dec_input: (batch_size=1, 1) int32 decoder initial input token (usually start token)
            partial: (batch_size=1, Tx=8) int32 partial decoder inputs for loop
        """

        X, enc_hidden, dec_input, partial = inputs

        # Run encoder
        enc_output, enc_hidden_new = self.encoder(X, enc_hidden)

        dec_hidden = enc_hidden_new  # initial decoder hidden state

        batch_size = tf.shape(X)[0]
        outputs = []

        # We perform decoding for each timestep t from 1 to Tx-1,
        # feeding previously generated tokens (from partial[:, t]) as decoder input.
        # This mimics teacher forcing / partial output feeding.
        d_i = dec_input  # initial decoder input for t=0

        for t in range(1, Tx):
            predictions, dec_hidden, _ = self.decoder(d_i, dec_hidden, enc_output)
            # The predictions shape is (batch_size, vocab_size)
            # Next decoder input is partial[:, t] reshaped to (batch_size,1)
            d_i = tf.reshape(partial[:, t], (batch_size, 1))

        # Final step prediction
        predictions, dec_hidden, _ = self.decoder(d_i, dec_hidden, enc_output)

        # For output, we take top 5 predicted token scores and indices from final predictions
        top_k = tf.math.top_k(predictions, k=5)  # values, indices

        # Output predictions and indices as tuple, matching the original model outputs
        return top_k.values, top_k.indices


def my_model_function():
    """
    Instantiate the MyModel class.
    """
    return MyModel()


def GetInput():
    """
    Returns a tuple of inputs suitable for MyModel call:
    - X: random integer sequence simulating encoder input (batch=1, Tx=8)
    - enc_hidden: initial encoder hidden state zeros (batch=1, units=1024)
    - dec_input: initial decoder input token IDs (batch=1, 1)
    - partial: partial decoder input sequences including start token (batch=1, Tx=8)
    
    All inputs data types and shapes designed to match MyModel expectations.
    """
    batch_size = 1

    # Encoder input: batch of sequences of token ids in [0,vocab_size)
    X = tf.random.uniform((batch_size, Tx), minval=0, maxval=vocab_size, dtype=tf.int32)

    # Encoder initial hidden state zeros, float32
    enc_hidden = tf.zeros((batch_size, units), dtype=tf.float32)

    # Decoder initial input token (e.g., start token id=1)
    dec_input = tf.ones((batch_size, 1), dtype=tf.int32)

    # Partial inputs for decoder for time steps (random valid token ids)
    partial = tf.random.uniform((batch_size, Tx), minval=0, maxval=vocab_size, dtype=tf.int32)

    return (X, enc_hidden, dec_input, partial)


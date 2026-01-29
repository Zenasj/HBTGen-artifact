# tf.random.uniform((BATCH_SIZE, input_length), dtype=tf.int32) ← Example input shape for the encoder inputs

import tensorflow as tf

# We build a fused MyModel combining Encoder and Decoder with attention, structured for TFLite conversion,
# adopting the approach from the issue where encoder and decoder are called as separate functions,
# and loops are avoided inside the model for TFLite compatibility.

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden size)
        # values shape: (batch_size, max_len, hidden size)
        # we expand query dims for broadcasting
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        # context vector is weighted sum of values
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, input_length):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.input_length = input_length

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length)
        # GRU with unroll=True to allow TFLite conversion (per discussion in issue)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       unroll=True)

    def call(self, x, hidden):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        output, state = self.gru(x, initial_state=hidden)
        # output shape: (batch_size, seq_length, enc_units)
        # state shape: (batch_size, enc_units)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       unroll=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # x shape after embedding: (batch_size, 1, embedding_dim)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        # concat context vector and embedding on last axis
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        # reshape output to (batch_size, vocab)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


# Fused model encapsulating Encoder and Decoder, exposing separated encoder and decoder inference steps
class MyModel(tf.keras.Model):
    def __init__(self,
                 vocab_inp_size,
                 vocab_tar_size,
                 embedding_dim=256,
                 units=1024,
                 batch_sz=64,
                 input_length=39):
        super(MyModel, self).__init__()
        self.batch_sz = batch_sz
        self.units = units
        self.input_length = input_length

        # Initialize Encoder and Decoder with unroll=True in GRU as per issue notes
        self.encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_sz, input_length)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_sz)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def encoder_infer(self, enc_input):
        # Initialize hidden state for encoder
        hidden = tf.zeros((tf.shape(enc_input)[0], self.units))
        enc_output, enc_hidden = self.encoder(enc_input, hidden)
        return enc_output, enc_hidden

    @tf.function(input_signature=[
        tf.TensorSpec([None, 1], tf.int32),       # dec_input (usually last predicted token)
        tf.TensorSpec([None, None, self.units], tf.float32),  # enc_output
        tf.TensorSpec([None, self.units], tf.float32)         # dec_hidden
    ])
    def decoder_infer(self, dec_input, enc_output, dec_hidden):
        # Run one decoding step
        predictions, dec_hidden_new, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)

        # Compute softmax scores explicitly (optional, for probabilities)
        scores = tf.nn.softmax(predictions, axis=1)

        # Greedy decode next token id
        predicted_ids = tf.expand_dims(tf.argmax(predictions, axis=1, output_type=tf.int32), 1)

        return predicted_ids, enc_output, dec_hidden_new, scores


def my_model_function():
    # Here we instantiate MyModel with typical NMT vocab sizes and input length from example:
    # Assuming from issue content vocab_inp_size ~ 40, vocab_tar_size ~ 7, batch_sz=64, input_length=39
    vocab_inp_size = 40
    vocab_tar_size = 7
    batch_sz = 64
    input_length = 39
    embedding_dim = 256
    units = 1024

    model = MyModel(vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_sz, input_length)
    return model


def GetInput():
    # Return a random integer input tensor for encoder input:
    # shape: (batch_size, input_length), dtype tf.int32
    batch_sz = 64
    input_length = 39
    # Values from 0 to 39 to simulate vocabulary indices (not guaranteed to match vocab, approximate)
    random_input = tf.random.uniform((batch_sz, input_length), minval=0, maxval=39, dtype=tf.int32)
    return random_input


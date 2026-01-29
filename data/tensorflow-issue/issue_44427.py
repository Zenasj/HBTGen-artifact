# tf.random.uniform((1, 64, 2048), dtype=tf.float32) ← input for encoder (batch=1, 64 features, feature_size=2048)
# tf.random.uniform((1, 1), maxval=5001, dtype=tf.int32) ← input token ids for decoder
# hidden state shape for decoder: (1, 512)

import tensorflow as tf

embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
features_shape = 2048  # feature size dimension from CNN encoder output

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch_size, 64, embedding_dim)
        # hidden: (batch_size, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch_size, 1, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  # (batch_size, 64, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, 64, 1)
        context_vector = attention_weights * features  # weighted features
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, embedding_dim)
        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Encoder expects raw CNN features of shape (batch, 64, features_shape)
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 64, features_shape), dtype=tf.float32)])
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       unroll=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 1), dtype=tf.int32, name='x'),
        tf.TensorSpec(shape=(1, 64, embedding_dim), dtype=tf.float32, name='features'),
        tf.TensorSpec(shape=(1, units), dtype=tf.float32, name='hidden')
    ])
    def call(self, x, features, hidden):
        # x: token id input (1,1)
        # features: encoder output (1, 64, embedding_dim)
        # hidden: hidden state (1, units)

        context_vector, attention_weights = self.attention(features, hidden)  # attention

        x = self.embedding(x)  # (1,1,embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # concat context vector with embedding (1,1,embedding_dim+embedding_dim)

        output, state = self.gru(x)  # GRU output and new state

        x = self.fc1(output)  # (1,1,units)
        x = tf.reshape(x, (-1, x.shape[2]))  # (1, units)
        x = self.fc2(x)  # (1, vocab_size)

        return x, state, attention_weights

    def reset_states(self, batch_size):
        return tf.zeros((batch_size, self.units))

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = vocab_size
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 64, features_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(1, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(1, units), dtype=tf.float32)
    ])
    def call(self, inputs):
        # inputs is a tuple: (encoder_input_features, decoder_input_token, decoder_hidden)
        encoder_input, dec_input_token, dec_hidden = inputs
        # Pass through encoder
        encoder_output = self.encoder(encoder_input)  # (1,64,embedding_dim)
        # Pass through decoder with encoder output
        pred, state, attention_weights = self.decoder(dec_input_token, encoder_output, dec_hidden)
        return pred, state, attention_weights

def my_model_function():
    return MyModel()

def GetInput():
    # Produce valid inputs for MyModel call:
    # encoder input: shape (1,64,2048) float32
    encoder_input = tf.random.uniform((1, 64, features_shape), dtype=tf.float32)
    # decoder input token ids: (1,1) int32, random int in [0, vocab_size)
    dec_input_token = tf.random.uniform((1,1), maxval=vocab_size, dtype=tf.int32)
    # initial hidden state: (1, units) zero or random float32
    dec_hidden = tf.zeros((1, units), dtype=tf.float32)
    return (encoder_input, dec_input_token, dec_hidden)


# tf.random.uniform((B, 150), dtype=tf.int32)
import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, lstm_units, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, name='ENCODE_BiLSTM_1'))
        self.lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=False, name='ENCODE_BiLSTM_2'))
        self.hidden = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu)

    def call(self, inputs, training=None):
        h = self.lstm1(inputs)
        h = self.lstm2(h)
        encoded = self.hidden(h)
        return encoded

class Decoder(tf.keras.layers.Layer):
    def __init__(self, max_len, lstm_units, vocab_size):
        super(Decoder, self).__init__()
        self.repeat = tf.keras.layers.RepeatVector(max_len)
        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, name='DECODE_BiLSTM_1'))
        self.lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, name='DECODE_BiLSTM_2'))
        self.out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(vocab_size, activation=tf.nn.softmax),
            name='OUTPUT')

    def call(self, inputs, training=None):
        h = self.repeat(inputs)
        h = self.lstm1(h)
        h = self.lstm2(h)
        output = self.out(h)
        return output

class Sentiment(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes):
        super(Sentiment, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, name='SENTIMENT_FF_1')
        self.senti_pred = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='SENTIMENT_OUT')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        out = self.senti_pred(x)
        return out

class MyModel(tf.keras.Model):
    def __init__(self,
                 lstm_units=64,
                 hidden_dim=50,
                 num_classes=2,
                 vocab_size=10000,
                 max_len=150,
                 embedding_dim=100):
        super(MyModel, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Modules
        self.embed_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, name='embedding')
        self.encoder_layer = Encoder(lstm_units, hidden_dim)
        self.sentiment_layer = Sentiment(hidden_dim, num_classes)
        self.decoder_layer = Decoder(max_len, lstm_units, vocab_size)

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, max_len), dtype int32
        embedded = self.embed_layer(inputs)  # (B, max_len, embedding_dim)
        encoded = self.encoder_layer(embedded, training=training)  # (B, hidden_dim)

        sentiment_pred = self.sentiment_layer(encoded, training=training)  # (B, num_classes)

        decoded = self.decoder_layer(encoded, training=training)  # (B, max_len, vocab_size)

        # Return tuple matching multiple outputs expected by Keras
        return decoded, sentiment_pred


def my_model_function():
    # Assumptions for parameters from original issue
    lstm_units = 64
    hidden_dim = 50
    num_classes = 2
    vocab_size = 10000
    max_len = 150
    embedding_dim = 100  # consistent with one-hot dimension in input generation

    return MyModel(lstm_units=lstm_units,
                   hidden_dim=hidden_dim,
                   num_classes=num_classes,
                   vocab_size=vocab_size,
                   max_len=max_len,
                   embedding_dim=embedding_dim)


def GetInput():
    # Produce random integer tensor of shape (batch_size, max_len)
    # batch_size chosen arbitrarily to 64 for typical usage
    batch_size = 64
    max_len = 150
    vocab_size = 10000

    # Random int input in [1, vocab_size) to simulate tokenized input sequences
    x = tf.random.uniform((batch_size, max_len), minval=1, maxval=vocab_size, dtype=tf.int32)
    return x


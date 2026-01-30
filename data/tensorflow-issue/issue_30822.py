import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

db = tf.data.Dataset.from_tensor_slices((x, {'decoded_mean': x_one_hot, 'pred': y}))
db = db.shuffle(1000).batch(batch_size)
auotencoder.fit(db, epochs=1)

batch_size = 64
total_words = 10000
max_length = 150
embedding_dim = 100

(x, y), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)

x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
print(x.shape, y.shape, x_test.shape, y_test.shape)

temp = np.zeros((x.shape[0], max_length, total_words))
temp[np.expand_dims(np.arange(x.shape[0]), axis=0).reshape(x.shape[0], 1), np.repeat(
    np.array([np.arange(max_length)]), x.shape[0], axis=0), x] = 1
x_one_hot = temp

# create dataset
db = tf.data.Dataset.from_tensor_slices((x, {'decoded_mean': x_one_hot, 'pred': y}))
db = db.shuffle(1000).batch(batch_size)

class Encoder(tf.keras.layers.Layer):

    def __init__(self, lstm_units, hidden_dim):
        super(Encoder, self).__init__()

        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units,
                                                                        return_sequences=True,
                                                                        name='ENCODE_BiLSTM_1'))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units,
                                                                        return_sequences=False,
                                                                        name='ENCODE_BiLSTM_2'))
        self.hidden = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu)

    def call(self, inputs, training=None):
        h = self.lstm1(inputs)
        h = self.lstm2(h)
        encoded = self.hidden(h)

        return encoded


class Decoder(tf.keras.layers.Layer):

    def __init__(self, max_len, lstm_units, vocab_size):
        super(Decoder, self).__init__()

        # [None, hidden_dim] -> [None, total_words, hidden_dim]
        self.repeat = tf.keras.layers.RepeatVector(max_len)

        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units,
                                                                        return_sequences=True,
                                                                        name='DECODE_BiLSTM_1'))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units,
                                                                        return_sequences=True,
                                                                        name='DECODE_BiLSTM_2'))
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size,
                                                                         activation=tf.nn.softmax),
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


def AE(lstm_units, hidden_dim, num_classes, vocab_size, max_len):
    input1 = tf.keras.layers.Input(shape=(max_len,), name='ENCODER_INPUT')

    embed_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    encoder_layer = Encoder(lstm_units, hidden_dim)
    sentiment_layer = Sentiment(hidden_dim, num_classes)
    decoder_layer = Decoder(max_len, lstm_units, vocab_size)

    embedded = embed_layer(input1)
    encoded = encoder_layer(embedded)

    sentiment_pred = sentiment_layer(encoded)

    decoded = decoder_layer(encoded)

    autoencoder = tf.keras.Model(inputs=input1, outputs=[decoded, sentiment_pred])

    return autoencoder

lstm_units = 64
num_classes = 2
hidden_dim = 50

autoencoder = AE(lstm_units, hidden_dim, num_classes, total_words, max_length)

autoencoder.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                    loss=[tf.losses.SparseCategoricalCrossentropy(), tf.losses.BinaryCrossentropy()],
                    metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(db, epochs=1) # this quit without showing any error
autoencoder.fit(x=x, y={'decoded_mean': x_one_hot, 'pred': y}, epoch=1) # this returns value error
autoencoder.fit(x=x, y=[x_one_hot, y], epoch=1) # this works
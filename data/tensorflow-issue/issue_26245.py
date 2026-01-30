from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import multi_gpu_model

vocab_size= 20000
maxlen=80

(X_train, y_train), (X_test, y_test) = \
    imdb.load_data(num_words=vocab_size)

X_train_pad = pad_sequences(X_train, maxlen=maxlen)
X_test_pad = pad_sequences(X_test, maxlen=maxlen)

with tf.device('/cpu:0'):
    model = Sequential([
        Embedding(vocab_size, 100, input_length=maxlen),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

model = multi_gpu_model(model, 2)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train_pad, y_train,
          batch_size=2048,
          epochs=2,
          shuffle=True)
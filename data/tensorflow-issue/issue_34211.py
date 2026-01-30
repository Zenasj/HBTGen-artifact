from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tf.keras.models.save_model(model, 
                          'imdb_model', 
                           include_optimizer=True, 
                           save_format='tf')

import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

(x_train, y_train), _ = imdb.load_data(num_words=20000)
x_train = sequence.pad_sequences(x_train, maxlen=80)

model = tf.keras.models.load_model('imdb_model')

model.fit(x_train, y_train, epochs=1)
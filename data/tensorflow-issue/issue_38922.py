from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

max_features = 20000
maxlen = 80

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_sample = x_train[:1]
a = model(x_sample, training=True)
dp_mask1 = model.layers[1].cell.get_dropout_mask_for_cell(x_sample, training=True, count=4)
rec_dp_mask1 = model.layers[1].cell.get_dropout_mask_for_cell(x_sample, training=True, count=4)

b = model(x_sample, training=True)
dp_mask2 = model.layers[1].cell.get_dropout_mask_for_cell(x_sample, training=True, count=4)
rec_dp_mask2 = model.layers[1].cell.get_dropout_mask_for_cell(x_sample, training=True, count=4)

# check if masks are the same after call
print(np.all([np.all(dp_mask1[i] == dp_mask2[i]) for i in range(len(dp_mask1))]))
print(np.all([np.all(rec_dp_mask1[i] == rec_dp_mask2[i]) for i in range(len(rec_dp_mask1))]))
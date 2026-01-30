model = Sequential()
model.add(layers.Input(shape=(None, 512), ragged=True))
model.add(layers.LSTM(32, return_sequences=True, dropout=0.4))
model.add(layers.TimeDistributed(layers.Dense(13, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np

x = tf.RaggedTensor.from_row_splits(np.ones((100, 512)), [0, 4, 20, 100])
y = tf.RaggedTensor.from_row_splits(np.ones((100, 13)), [0, 4, 20, 100])
print(x.shape)
print(y.shape)
model = Sequential()
model.add(layers.Input(shape=(None, 512), ragged=True))
model.add(layers.LSTM(32, return_sequences=True, dropout=0.4))
model.add(layers.TimeDistributed(layers.Dense(13, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
history = model.fit(x=x, y=y, epochs=10, verbose=1)
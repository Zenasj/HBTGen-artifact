model = Sequential()
model.add(layers.TimeDistributed(layers.Masking(-1),input_shape=(None,20,1)))
model.add(layers.TimeDistributed(layers.LSTM(num_units_1,dropout=0.4)))
model.add(layers.LSTM(num_units_2))
model.add(layers.Dense(1))
model.summary()

history = model.fit(train_data, epochs=epochs, verbose=1, steps_per_epoch=-(-sample_count//batch_size))

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np

x = tf.RaggedTensor.from_row_splits(np.ones((100,20,1)),[0,4,20,100])
y = np.ones((3,1))

model = Sequential()
model.add(layers.TimeDistributed(layers.LSTM(32,dropout=0.4),input_shape=(None,20,1)))
model.add(layers.LSTM(24))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
history = model.fit(x=x, y=y, epochs=10, verbose=1)
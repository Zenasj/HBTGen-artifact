import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout
import numpy as np

num_features = 205
time_lenth = 12
num_of_instances = 5000
trip_sets = np.random.rand(num_of_instances, time_lenth, num_features)
print('num features: ', num_features)
data_len = len(trip_sets)
test_split = np.arange(data_len)
np.random.shuffle(test_split)
new_dataset = np.array(trip_sets)
targets = np.random.rand(num_of_instances, time_lenth, 1)

test_data = new_dataset[test_split[:int(data_len*0.2)]]
y_test_data = targets[test_split[:int(data_len*0.2)]]
train_data = new_dataset[test_split[int(data_len*0.2):]]
y_train_data = targets[test_split[int(data_len*0.2):]]

model = Sequential()
model.add(LSTM(75, return_sequences=True, input_shape=(None, num_features)))
model.add(Dropout(0.3))
model.add(LSTM(75, return_sequences=True))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(1)))
# Memory leak also occurs if i use a model.add(Dense(1)) below instead of Time Distributed
# model.add(TimeDistributed(Dense(1)))

adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)
history = model.fit(x=train_data, y=y_train_data, epochs=100, validation_data=(test_data,y_test_data))
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# -*- coding: utf-8 -*-
# @author = Neel Gupta

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as pyplot
import csv

def csv_arr(csv_f):
    results = []
    # the loop..
    with open(csv_f) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    # returning the output
    return np.asarray(results)

train_data = np.reshape(csv_arr("train.csv"), (15,2))
test_data = csv_arr("test.csv")

x_data = tf.keras.Input(shape=(15,2))

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, )

# fit model
history = model.fit(train_data, epochs=100, verbose=1)

# getting model's summary
model.summary()

# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
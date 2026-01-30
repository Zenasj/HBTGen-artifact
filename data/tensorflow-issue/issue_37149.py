from tensorflow.keras import models
from tensorflow.keras import optimizers

import requests
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, date

#split the data into x-train and y-train datasets
x_train = []
y_train = []

for i in range(20, len(train_data)):
    x_train.append(train_data[i-20:i, 0])
    y_train.append(train_data[i, 0])

    if i<= 20:
        print(x_train)
        print(y_train)

#convert x-train and y-train to numpy arrays to train models
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data, LSTM model expects 3D dataset
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build LSTM MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=1)

#create testing data set
#creat new array containing scaled values
test_data = scaled_data[training_data_len - 20: , :]

#create the datasets x-test and y-test
x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(20, len(test_data)):
    x_test.append(test_data[i-20:i, 0])

#convert data to numpy array
x_test = np.array(x_test)

#reshape data to 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get root mean squared error
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

#get closing price
data = df.filter(['Close'])

#get closing price values
dataset = data.values

#set training data length to len of total data set
training_data_len = math.ceil(len(dataset))
print(training_data_len)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

for i in range(20, len(test_data)+1):
    x_test.append(test_data[i-20:i, 0])
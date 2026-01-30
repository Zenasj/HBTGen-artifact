import random
from tensorflow.keras import layers
from tensorflow.keras import models

# Imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import load_model



# Create dummy dataset
dummydata = pd.DataFrame(np.random.random_sample((10000, 51)))

predictors_list = dummydata.columns.tolist()
predictors_list.remove(predictors_list[-1])

X_rgs = dummydata[predictors_list]
y_rgs = dummydata[dummydata.columns[-1]]




# Train, Test split
train_length = int(len(dummydata)*0.70)
X_rgs_train = X_rgs[:train_length]
X_rgs_test = X_rgs[train_length:]
y_rgs_train = y_rgs[:train_length]
y_rgs_test = y_rgs[train_length:]




# pandas to numpy
X_rgs_train = X_rgs_train.to_numpy()
X_rgs_test = X_rgs_test.to_numpy()



# Reshape data
y_rgs_train = y_rgs_train.values
y_rgs_train = y_rgs_train.reshape(len(y_rgs_train), 1)
y_rgs_train = np.ravel(y_rgs_train)

y_rgs_test = y_rgs_test.values
y_rgs_test = y_rgs_test.reshape(len(y_rgs_test), 1)
y_rgs_test = np.ravel(y_rgs_test)


X_train_lstm = []
y_train_lstm = []

for i in range(60, X_rgs_train.shape[0]):
    X_train_lstm.append(X_rgs_train[i-60:i])
    y_train_lstm.append(X_rgs_train[i, 0])


len_y_train_lstm = len(y_train_lstm) 
y_train_lstm = y_rgs_train[:len_y_train_lstm]
X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)


X_test_lstm = []
y_test_lstm = []

for i in range(60, X_rgs_test.shape[0]):
    X_test_lstm.append(X_rgs_test[i-60:i])
    y_test_lstm.append(X_rgs_test[i, 0])

len_y_test_lstm = len(y_test_lstm) 
y_test_lstm = y_rgs_test[:len_y_test_lstm]
X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)





# Model architecture and vars
EPOCHS = 1 
BATCH_SIZE = 32 

model = Sequential()
model.add(LSTM(128, input_shape=(X_train_lstm.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization()) 

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])




model.fit(X_train_lstm, y_train_lstm, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test_lstm, y_test_lstm)) # fit
model.save("model_ISSUE.h5") # save entire model


del model 


model = load_model('model_ISSUE.h5') # load entire model
model.fit(X_train_lstm, y_train_lstm, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test_lstm, y_test_lstm)) # fit
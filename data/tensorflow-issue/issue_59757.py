from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_addons.layers import ESN
from tensorflow_addons.rnn import ESNCell
from tensorflow.keras.layers import RNN
from tensorflow.keras.layers import SimpleRNN, SimpleRNNCell
from sklearn.preprocessing import MinMaxScaler
from tensorflow import random as rnd


# Fix the seed
rnd.set_seed(0)


# The data can be downloaded from https://mantas.info/wp/wp-content/uploads/simple_esn/MackeyGlass_t17.txt
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data.reshape(-1, 1))

# Split Dataset in Train and Test
train, test = scaled[0:-100], scaled[-100:]

# Split into input and output 
train_X, train_y = train[:-1], train[1:]
test_X, test_y = test[:-1], test[1:] 

# Reshaping 
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Batch and epochs
batch_size = 20
epochs = 3


# Design and run the model
model = Sequential()
model.add(RNN(SimpleRNNCell(1)))
#model.add(ESN(units = 12, spectral_radius = spectral_radius, leaky=0.75, connectivity = 0.9)) # this line works exactly like the next one
#model.add(RNN(ESNCell(12, spectral_radius = spectral_radius, leaky=0.75, connectivity = 0.9)))
model.add(Dense(train_y.shape[1]))
model.compile(loss='huber', optimizer='adam')

model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# Print the weights of the dense layer
#print(model.layers[1].get_weights())
#for layer in model.layers: print(layer.get_config(), layer.get_weights())
for layer in model.layers: print(layer.get_weights())
import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	print('Could not set GPU memory growth')
	pass
	
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


N = 1000
T = 2000
C = 5
batch_size = 1024

x = np.zeros((T+1, N, C), dtype='float32')

for t in range(T+1):
	if t == 0:
		x[t,:,] = np.random.normal(0., 1., (N, C))
	else:
		x[t,:,] = 0.95 * x[t-1,:,] + 0.05 * np.random.normal(0., 1., (N, C))

x = np.swapaxes(x, 0, 1)

for i in range(100):
	keras.backend.clear_session()
	input_layer = keras.layers.Input(shape=(None, C))
	nn = keras.layers.LSTM(batch_input_shape=(batch_size, T, C), units=32, return_sequences=True)(input_layer)
	nn = keras.layers.LSTM(units=16, return_sequences=True)(nn)
	nn = keras.layers.LSTM(units=8)(nn)
	output_layer = keras.layers.Dense(1)(nn)
	model = keras.Model(input_layer, output_layer)
	opt = keras.optimizers.Adam(lr=0.001)
	model.compile(loss='mse', optimizer=opt)

	c = EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True)
	model.fit(x[:750,:-1,:], x[:750,-1,1], batch_size=128, callbacks=[c],
		validation_data=(x[750:,:-1,:], x[750:,-1,1]), epochs=100)
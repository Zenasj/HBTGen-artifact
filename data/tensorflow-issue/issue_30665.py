import random
from tensorflow.keras import layers
from tensorflow.keras import models

left = Input(shape=(128, 3072), dtype='float32', name='Input-Left')
right = Input(shape=(128, 3072), dtype='float32', name='Input-Right')
lstm = Bidirectional(LSTM(units=768,
                          activation='tanh'),
                      name='Bidirectional-LSTM')
l_lstm = lstm(left)
r_lstm = lstm(right)
subtracted = Subtract(name='Subtract')([l_lstm, r_lstm])
abs_subtracted = Lambda(function=backend.abs)(subtracted)
mul = Multiply(name='multiplication')([l_lstm, r_lstm])
concat = concatenate([abs_subtracted, mul])
output = Dense(units=1)(concat)
model = Model(inputs=[left, right],
              outputs=output)
model = multi_gpu_model(model, gpus=2)
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['acc'])
import numpy as np
x1 = np.random.rand(100, 128, 3072)
x2 = np.random.rand(100, 128, 3072)
y = np.random.rand(100)
model.fit(x = [x1, x2],
         y=y,
         epochs=10)

import numpy as np
from keras.layers import Input, Bidirectional, LSTM, Subtract, Lambda, Multiply, concatenate, Dense
from keras.utils import multi_gpu_model
from keras.models import Model
import keras.backend as backend

x1 = np.random.rand(100, 128, 3072)
x2 = np.random.rand(100, 128, 3072)
y = np.random.rand(100)

left = Input(shape=(128, 3072), dtype='float32', name='Input-Left')
right = Input(shape=(128, 3072), dtype='float32', name='Input-Right')
lstm = Bidirectional(LSTM(units=768,
                          activation='tanh'),
                     name='Bidirectional-LSTM')
l_lstm = lstm(left)
r_lstm = lstm(right)
subtracted = Subtract(name='Subtract')([l_lstm, r_lstm])
abs_subtracted = Lambda(function=backend.abs)(subtracted)
mul = Multiply(name='multiplication')([l_lstm, r_lstm])
concat = concatenate([abs_subtracted, mul])
output = Dense(units=1)(concat)
model = Model(inputs=[left, right],
              outputs=output)
model = multi_gpu_model(model, gpus=2)
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['acc'])
model.fit(x=[x1, x2],
          y=y,
          epochs=10)
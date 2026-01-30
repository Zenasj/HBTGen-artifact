from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import time

feature_dimension=2
lstm_size=256
learning_rate=0.0001

model = tf.keras.Sequential([
    tf.keras.Input([None, feature_dimension]),
    tf.keras.layers.LSTM(lstm_size),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['mse', 'mae'])

print('Model.predict')
st = time.time()
for i in range(5,100):
    model.predict(np.array([[(0.1,0.1)]*i]*10))
print(f'Model.predict in {time.time() - st}')

print('Model.predict_on_batch')
st = time.time()
for i in range(5,100):
    np.array(model.predict_on_batch(np.array([[(0.1,0.1)]*i]*10)))

print(f'Model.predict_on_batch in {time.time() - st}')
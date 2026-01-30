import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

data=np.random.rand(30,16000)
data = np.expand_dims(data, axis=2)
#model = tf.keras.models.load_model('newmodel.h5')
model = keras.Sequential()
model.add(keras.layers.LSTM(15, input_shape=(16000, 1), return_sequences=True))
for i in range(8):
    model.add(keras.layers.LSTM(15, return_sequences=True))
model.add(keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
est=model.predict(data)
model.save("newmodel.h5")

data=np.random.rand(30,16000)
data = np.expand_dims(data, axis=2)
model = tf.keras.models.load_model('newmodel.h5')
est=model.predict(data)
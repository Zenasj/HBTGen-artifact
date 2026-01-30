import random
from tensorflow.keras import layers
from tensorflow.keras import models

# previously saved model with tf 2.14
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np

# # print version
# print(tf.__version__)

# # random data
# data = np.random.random((100, 10))
# labels = np.random.random((100, ))

# # simple model
# normalizer = keras.layers.Normalization()
# normalizer.adapt(data)
# model = keras.Sequential()
# model.add(normalizer)
# model.add(keras.layers.Dense(10))
# model.add(keras.layers.Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(data, labels, epochs=10, verbose=2)
# model.save('test.keras')


### loading with tf 2.15 ###
import tensorflow as tf
from tensorflow import keras

# print version
print(tf.__version__)

# load model
model = keras.models.load_model('test.keras')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda,Flatten

def dummy_padding(x, padding_size):
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')

def dummy_nn(n_hidden, kernel_size=28, padding_size=5):
    model = Sequential()
    model.add(Lambda(lambda x: dummy_padding(x, padding_size), input_shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(10))
    return model

model_ = dummy_nn(500)

tf.keras.models.save_model(model_, 'model_.h5')
model_.summary()

import tensorflow as tf
model2_ = tf.keras.models.load_model('model_.h5')
model2_.summary()

import tensorflow as tf

def dummy_padding(x, padding_size):
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='SYMMETRIC')

model2_ = tf.keras.models.load_model('model_.h5')
model2_.summary()
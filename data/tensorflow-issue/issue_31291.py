import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Embedding, Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import Model
import numpy as np
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_more():
    while(True):
        yield ({'a_input': np.random.randint(0, 10, (32, 1200))},
               np.random.rand(32, 1))


def build():

    input = Input(shape=(1200,), name='a_input', dtype='int32')
    x = Embedding(input_dim=10,
                  output_dim=4,
                  input_length=1200,
                  trainable=True, name='embedding')(input)
    x = Dense(1, activation='linear')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-1))(x)
    x = Dense(1, name='output')(x)
    this_model = Model(input, x)

    return this_model


####### FAIL
# Situation 1 (+METRICS +MOMENTUM)
#######
K.clear_session()
this_model = build()
optimizer = keras.optimizers.SGD(lr=0.05, momentum=0.9)
this_model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
this_model.fit_generator(iter(get_more()), steps_per_epoch=10)

####### PASS
# Situation 2 (+METRICS -MOMENTUM)
#######
K.clear_session()
this_model = build()
optimizer = keras.optimizers.SGD(lr=0.05)
this_model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
this_model.fit_generator(iter(get_more()), steps_per_epoch=10)

####### PASS
# Situation 3 (-METRICS +MOMENTUM)
#######
K.clear_session()
this_model = build()
optimizer = keras.optimizers.SGD(lr=0.05, momentum=0.9)
this_model.compile(loss='mse', optimizer=optimizer)
this_model.fit_generator(iter(get_more()), steps_per_epoch=10)

####### PASS
# Situation 4 (+METRICS +MOMENTUM)  (fit instead of fit_generator)
#######
K.clear_session()
this_model = build()
optimizer = keras.optimizers.SGD(lr=0.05, momentum=0.9)
this_model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
x_in = {'a_input': np.random.randint(0,10,(500,1200))}
y_out = np.random.rand(500,1)
this_model.fit(x_in, y_out)

with tf.device('/cpu:0'):
  this_model.fit_generator(iter(get_more()), steps_per_epoch=10)
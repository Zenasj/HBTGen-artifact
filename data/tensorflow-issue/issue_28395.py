from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

window_size = 1024 #900
inputs_n = 7
outputs_n = 8
epochs_n = 1
epochs =  range(epochs_n)

TPU_ADDRESS = 'grpc://10.240.1.2:8470'

from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model

model = Sequential()
model.add(LSTM(128, batch_input_shape=(1, window_size, inputs_n), return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(Dense(8, activation='linear'))

opt = tf.train.AdamOptimizer(0.01)

model.compile(loss='mse', optimizer=opt)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)))
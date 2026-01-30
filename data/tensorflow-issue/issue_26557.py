import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization, Bidirectional,
                                     Conv1D, Dense, Dropout, Input, Lambda, Masking,
                                     TimeDistributed)

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"


Input_layer = Input(shape=(2, ))
Dense1 = Dense(8, input_shape=(2, ))(Input_layer)
Dense1_stop = Lambda(lambda x: tf.stop_gradient(x))(Dense1)
Dense2 = Dense(4)(Dense1_stop)
Dense3 = Dense(1, activation='softmax')(Dense2)
model = Model(inputs=Input_layer, outputs=Dense3)

# model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse')
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')


x = np.random.uniform(0, 1, (100, 2))
y = np.random.uniform(0, 1, (100, 1))
model.fit(x=x, y=y, validation_split=0.2)
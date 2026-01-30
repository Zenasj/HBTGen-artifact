import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

input = tf.keras.layers.Input(shape=(20,))
output = tf.keras.layers.Dense(2)(input)
model = tf.keras.models.Model(inputs=input, outputs=output)
model.compile(loss='mse', optimizer='sgd')

model.fit(np.random.normal(0, 1, (200, 20)), np.random.normal(0, 1, (200, 2)))

# coding: utf-8
import tensorflow as tf
import os

if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Only use CPU:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
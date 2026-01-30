import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


features = np.random.uniform(size=[100, 50])
labels = np.random.uniform(size=[100])

linear_model = tf.keras.experimental.LinearModel()
linear_model.compile('adagrad', 'mse')
linear_model.fit(features, labels)
dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
dnn_model.compile('rmsprop', 'mse')
dnn_model.fit(features, labels)
combined_model = tf.keras.experimental.WideDeepModel(dnn_model, linear_model)
# Uncomment the following line to test without compilation:
combined_model.compile(optimizer=['adagrad', 'rmsprop'], loss='mse')
combined_model.fit(features, labels)
tf.saved_model.save(combined_model, "/tmp/saved_model")
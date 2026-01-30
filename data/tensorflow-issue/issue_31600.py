from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

# suppose batch_size=1
# trainarray is a 2-D array of shape [1, 7] of batch_size=1
trainarray = np.array([[0,0,0,0,0,0,0]])
# excluding batch_size
# for more details, please refer to the first example in
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
input_shape = [trainarray.shape[1]]

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(7, input_shape=input_shape),
  tf.keras.layers.Dense(12),
  tf.keras.layers.Dense(1)
])

model.predict(np.array([[0,0,0,0,0,0,0]]))
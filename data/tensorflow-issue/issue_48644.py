from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.keras.backend.set_floatx('float64')
model = keras.Sequential([
keras.layers.AveragePooling3D(pool_size=(3, 1, 3), input_shape=(3, 3, 3, 4))])
x = tf.constant([[[[[2, 2, 2, 1], [1, 1, 2, 2], [1, 1, 1, 1]], [[1, 1, 1, 2], [2, 1, 2, 2], [1, 2, 2, 2]], [[1, 1, 1, 2], [1, 1, 1, 1], [2, 1, 1, 1]]], [[[1, 2, 2, 1], [2, 2, 1, 1], [1, 2, 1, 1]], [[2, 2, 2, 1], [1, 1, 1, 1], [2, 2, 1, 2]], [[1, 2, 2, 1], [2, 2, 2, 1], [1, 1, 2, 1]]], [[[1, 1, 1, 1], [2, 2, 1, 2], [1, 1, 1, 2]], [[1, 1, 2, 1], [1, 1, 1, 1], [2, 2, 1, 2]], [[2, 2, 2, 1], [2, 1, 1, 2], [1, 1, 2, 2]]]]])
print (np.array2string(model.predict(x,steps=1), separator=', '))

import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.keras.backend.set_floatx('float64')
model = keras.Sequential([
keras.layers.MaxPool3D(pool_size=(1, 2, 2), input_shape=(4, 4, 3, 3))])
x = tf.constant([[[[[1, 2, 2], [1, 2, 2], [2, 2, 1]], [[1, 2, 1], [1, 2, 2], [1, 1, 2]], [[2, 2, 1], [2, 1, 1], [1, 2, 1]], [[2, 1, 2], [1, 1, 2], [1, 1, 1]]], [[[2, 1, 2], [2, 2, 1], [1, 2, 2]], [[1, 2, 2], [2, 1, 1], [2, 2, 2]], [[2, 1, 1], [2, 1, 2], [2, 1, 2]], [[2, 1, 2], [2, 2, 1], [1, 1, 2]]], [[[1, 1, 1], [2, 1, 2], [1, 2, 2]], [[2, 2, 1], [1, 2, 1], [1, 1, 1]], [[1, 1, 2], [2, 2, 2], [2, 2, 1]], [[1, 2, 1], [1, 1, 2], [1, 1, 2]]], [[[2, 1, 1], [1, 1, 1], [2, 1, 2]], [[1, 2, 1], [2, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 1, 2], [2, 2, 2]], [[2, 1, 1], [2, 1, 1], [1, 2, 1]]]]])
print (np.array2string(model.predict(x,steps=1), separator=', '))
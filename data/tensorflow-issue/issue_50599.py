import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
tf.keras.backend.set_floatx('float64')
model = keras.Sequential([
keras.layers.ConvLSTM2D(1, (1, 2), return_sequences=True, padding='valid', strides=(1,1),recurrent_activation='linear', activation='linear',  input_shape=(2, 1, 3, 1))])
w = model.get_weights()
w[0] = np.array([[[[1, 1, 1, 1]],[[1, 1, 1, 1]]]])
w[1] = np.array([[[[1, 1, 1, 1]],[[1, 1, 1, 1]]]])
w[2] = np.array([0, 0, 0, 0])
model.set_weights(w)
x = tf.constant([[[[[0], [0], [2]]], [[[0], [0], [1]]]]])
print (np.array2string(model.predict(x,steps=1), separator=', '))
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding))
model.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
model.fit(np.random.rand(10, 1, 1, 3))
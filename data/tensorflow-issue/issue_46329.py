import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# Code is from https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10, trainable=False))

input_array = np.random.randint(1000, size=(32, 10))
model.compile('rmsprop', 'mse')

model.summary()
model.save('some_path')
new_model = tf.keras.models.load_model('some_path')
new_model.summary()
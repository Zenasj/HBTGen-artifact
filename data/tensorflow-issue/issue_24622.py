from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, input_shape=[2]))
model.add(tf.keras.layers.Dense(1))
tf.keras.utils.plot_model(model, to_file='my_model.png')
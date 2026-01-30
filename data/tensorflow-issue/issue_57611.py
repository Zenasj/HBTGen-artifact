from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=1))
model.add(tf.keras.layers.ReLU())
model.compile(loss='MSE')

x = tf.ones(shape=(1, 32, 1))
model.fit(x, y=x, epochs=2)
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(1)
tf.set_random_seed(1)
# FEATURE REQUEST for another parameter like the ones above that we can set
# here and avoid repeating initializer seed in each and every layer below

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=1), input_shape=[1]))
model.add(tf.keras.layers.Dense(8, activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=1)))
model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.glorot_uniform(seed=1)))
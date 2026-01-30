import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

policy = tf.keras.mixed_precision.Policy("mixed_float16")
#runs fine with all the same type
#policy = tf.keras.mixed_precision.Policy("float16")

tf.keras.mixed_precision.set_global_policy(policy)

model = tf.keras.Sequential()

model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(10, activation="relu")))

model.compile(loss=tf.keras.losses.MeanSquaredError())

X = np.random.randn(1,20)
Y = np.random.randn(1,10)

model.fit(X,Y)
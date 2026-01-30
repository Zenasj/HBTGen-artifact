import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

seq1 = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3), tf.keras.layers.BatchNormalization(axis=3, momentum=0.0), tf.keras.layers.LeakyReLU(0.01)])

import numpy as np
x = np.random.randn(128,32,32,1)

res1 = seq1(x, training=True)
res2 = seq1(x, training=False)

print(np.linalg.norm(res1 - res2))
import random
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

import tensorflow as tf
A = tf.random.normal((5, 5))
b = tf.random.normal((5,1))
tf.linalg.solve(A,b)
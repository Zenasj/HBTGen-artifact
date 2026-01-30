import random
from tensorflow import keras

import tensorflow as tf
import numpy as np
# Build concat model and set input data type to match the new input data types downstream
x1 = tf.keras.Input([5], dtype='float64')
x2 = tf.keras.Input([6], dtype= 'float64')
assert x1.dtype == 'float64'
assert x2.dtype == 'float64'
y = tf.concat([x1, x2], axis=1)
concat_model = tf.keras.Model(inputs=[x1, x2], outputs=y)
# Generate inputs
x1_ = np.random.random(size=[2, 5])
x2_ = np.random.random(size=[2, 6])
assert x1_.dtype == 'float64'
assert x2_.dtype == 'float64'
# Run model
y_ = concat_model([x1_, x2_])
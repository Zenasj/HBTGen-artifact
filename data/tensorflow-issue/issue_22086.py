from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.python.ops import gen_linalg_ops

a = tf.placeholder(tf.float16, shape=[2, 2])
gen_linalg_ops.qr(a)

import tensorflow.keras as keras

model = keras.Sequential()
model.add(keras.layers.GRU(128, recurrent_dropout=0.75, input_shape=(200, 4)))
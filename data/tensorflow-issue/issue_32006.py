from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def cond(i, x):
    return tf.reduce_all(x < 10)

def body(i, x):
    return i + 1, x + i

x = tf.keras.layers.Input(shape=(), dtype=tf.float32)
inc = tf.while_loop(cond, body, [tf.constant(0, dtype=tf.float32), x])
# the following fixes things
# inc = tf.keras.layers.Lambda(lambda x: tf.while_loop(
#     cond, body, [tf.constant(0, dtype=tf.float32), x]))(x)

model = tf.keras.Model(inputs=x, outputs=inc)  # <- error occurs here
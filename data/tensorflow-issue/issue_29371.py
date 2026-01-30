import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
inps = tf.keras.Input(shape=(None, 256), name="inps")
mask = tf.keras.Input(shape=(1, 1, None), name="mask")
m1 = tf.random.uniform(shape=(8, 20))
m2 = tf.random.uniform(shape=(8, 20))
outputs = tf.keras.layers.Dense(units=512)(m1 + m2)
model = tf.keras.Model(inputs=[inps, mask], outputs=outputs, name="test")
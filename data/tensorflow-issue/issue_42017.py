from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

inp1 = tf.keras.Input(1)
h1 = tf.keras.layers.Dense(1)(inp1)
out1 = tf.keras.layers.Dense(1)(h1)
model1 = tf.keras.Model(inputs=inp1, outputs=out1)

inp2 = tf.keras.Input(1)
h2 = model1(inp2)
out2 = tf.keras.layers.Dense(1)(h2)
model2 = tf.keras.Model(inputs=inp2, outputs=[out2, model1.layers[1].output])
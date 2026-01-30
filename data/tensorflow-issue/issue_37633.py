import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

python
import tensorflow as tf

inp = tf.keras.Input((None, 3))
inp.set_shape((None, 2, 3))
x = tf.keras.layers.Dense(3)(inp)

model = tf.keras.Model(inp, x)
model.summary()

model.save("test.h5")
loaded = tf.keras.models.load_model("test.h5")
loaded.summary()

python
import tensorflow as tf

inp = tf.keras.Input((2, 3))
x = tf.zeros(tf.shape(inp)[:2])

model = tf.keras.models.Model(inp, x)
model.summary()
model.save("test.h5")
loaded = tf.keras.models.load_model("test.h5")
loaded.summary()

python
inp = tf.keras.Input((3, 4))
x = ...  # many layers, output shape is (None, None, None)
x.set_shape((None, 2, 3))
x = tf.keras.layers.Dense(3)(x)
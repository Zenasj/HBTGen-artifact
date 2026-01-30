from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["a", "b"])

s = tf.keras.Input(1, dtype=tf.string)
k = lookup(s)

model = tf.keras.Model(s, k)
model.compile()

tf.keras.models.save_model(model, "./out/dummy/")

load2 = tf.keras.models.load_model("./out/dummy/")
tf.keras.models.save_model(load2, "./out/dummy2/")

load3 = tf.keras.models.load_model("./out/dummy2/")
tf.keras.models.save_model(load3, "./out/dummy3/")

import tensorflow as tf

lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["a", "b"])

s = tf.keras.Input(1, dtype=tf.string)
k = lookup(s)

model = tf.keras.Model(s, k)
model.compile()

tf.keras.models.save_model(model, "/tmp/dummy/")

load2 = tf.keras.models.load_model("/tmp/dummy/")
tf.keras.models.save_model(load2, "/tmp/dummy2/")

load3 = tf.keras.models.load_model("/tmp/dummy2/")
tf.keras.models.save_model(load3, "/tmp/dummy3/")
from tensorflow.keras import layers
from tensorflow.keras import models

x = keras.Input(shape=1, name="x")
ones_like_layer = keras.layers.Lambda(K.ones_like, name="ones_like")
ones_like_layer(x)
logits = keras.layers.Dense(1, activation="sigmoid")

model = keras.Sequential([x, ones_like_layer, logits], name="ones_like_model")

class OnesLike(keras.layers.Layer):
  def call(self, x):
    return K.ones_like(x)

import tensorflow as tf
from tensorflow import keras
x = keras.layers.Input(shape=(1), name="input_layer")
ones_like_layer = keras.layers.Lambda(lambda x: tf.ones_like(x), name="ones_like")
ones_like_layer(x)
logits = keras.layers.Dense(1, activation="sigmoid")
model = keras.Sequential([x, ones_like_layer, logits], name="ones_like_model")
keras.models.model_from_json(model.to_json())

ones_like_layer = keras.layers.Lambda(lambda x: K.ones_like(x), name="ones_like")

ones_like_layer = keras.layers.Lambda(K.ones_like, name="ones_like")
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras
from keras import layers

inp = keras.layers.Input(shape=(1,), name="input")
result = keras.layers.Dense(250, name="dense_layer")(inp)

model = keras.models.Model(inp, result, name="mdl")
loss = "categorical_crossentropy"
used_optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=loss, optimizer=used_optimizer)

model.save("test-model.keras")
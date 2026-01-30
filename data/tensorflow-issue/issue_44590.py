import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

layer_units = [2, 1]

tf.random.set_seed(0)
model = tf.keras.Sequential()
inputs = tf.keras.Input(shape=(1,))
model.add(inputs)
layer1 = tf.keras.layers.Dense(layer_units[0])
layer1.add_loss(lambda :tf.reduce_sum(layer1.kernel))
model.add(layer1)
layer2 = tf.keras.layers.Dense(layer_units[1])
layer2.add_loss(lambda :tf.reduce_sum(layer2.kernel))
model.add(layer2)

print(model.losses)

layer_units = [2, 1]

tf.random.set_seed(0)
model = tf.keras.Sequential()
inputs = tf.keras.Input(shape=(1,))
model.add(inputs)
for i in range(2):
    layer = tf.keras.layers.Dense(layer_units[i])
    layer.add_loss(lambda :tf.reduce_sum(layer.kernel))
    model.add(layer)

print(model.losses)

for i in range(2):
    layer = tf.keras.layers.Dense(1)
    layer.add_loss(lambda :tf.reduce_sum(layer.kernel))
    model.add(layer)

def make_loss(layer):
    return lambda : tf.reduce_sum(layer.kernel)

for i in range(2):
    layer = tf.keras.layers.Dense(1)
    layer.add_loss(make_loss(layer))
    model.add(layer)
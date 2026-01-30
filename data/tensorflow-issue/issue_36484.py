import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

input = tf.keras.layers.Input(shape=(28,28))

x = tf.keras.layers.Flatten()(input)
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dense(10)(x)

small_model = tf.keras.models.Model(inputs=[input], outputs=[x])

y = small_model(input)

y = tf.keras.layers.Dense(20)(y)
y = tf.keras.layers.Dense(30)(y)
y = tf.keras.layers.Dense(10)(y)

big_model = tf.keras.models.Model(inputs=[small_model.input], outputs=[y])

ext_model = tf.keras.models.Model(inputs=[small_model.input], outputs=[big_model.output,  big_model.layers[1].output])

images = tf.random.uniform(shape=(32,28,28))
labels = tf.zeros(shape=(32,))

with tf.GradientTape() as tape:    
    predictions, layer = ext_model(images)

print("grad = ", tape.gradient(predictions, layer))

ext_model = tf.keras.models.Model(inputs=[small_model.input], outputs=[big_model.output,  big_model.layers[2].output])
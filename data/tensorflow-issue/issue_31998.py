import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model as M
from tensorflow.keras import Input as I

x = tf.cast(np.random.randn(1, 100, 100), tf.float32)
y = tf.cast(np.random.randn(1, 100), tf.float32)
z = tf.cast(np.random.randn(1, 100), tf.float32)
u = tf.cast(np.random.randn(1, 100), tf.float32)

class Model(M):
    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.GRU(100)
        self(I(shape=(None, 100)), I(shape=(100,)))

    @tf.function # remove this and it works fine
    def call(self, x, y):
        z = self.layer(x, initial_state=y)
        return z


model = Model()

with tf.GradientTape(persistent=True) as tape: # if persistent=False it works fine
    loss = tf.norm(model(x, y) - z)
grads = tape.gradient(loss, model.trainable_variables)

print("###############SUCCESS################")

class Model2(M):
    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.LSTM(100)
        self(I(shape=(None, 100)), I(shape=(100,)), I(shape=(100,)))

    @tf.function     # remove this and it works fine
    def call(self, s, h, c):
        z = self.layer(s, initial_state=(h, c))
        return z


model2 = Model2()

with tf.GradientTape(persistent=True) as tape: # if persistent=False it works fine
    loss = tf.norm(model2(x, y, u) - z)
grads = tape.gradient(loss, model2.trainable_variables)


print("###############SUCCESS################")

self.layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100))
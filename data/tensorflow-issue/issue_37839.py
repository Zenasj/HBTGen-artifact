import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

class TestModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


model = TestModel()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',      
              metrics=['mae'])  

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(
    data, labels, epochs=10, batch_size=32,
    callbacks=[tf.keras.callbacks.ModelCheckpoint('model_{epoch:02d}.ckpt',
                                                  save_weights_only=False)])

dense1 = tf.keras.layers.Dense(64, activation='relu')
dense2 = tf.keras.layers.Dense(64, activation='relu')
dense3 = tf.keras.layers.Dense(10)
inputs = tf.keras.Input((32,), dtype='float32')
output = dense3(dense2(dense1(inputs)))
model = tf.keras.Model(inputs, output)

class TestModel(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

inputs = tf.keras.Input((32,), dtype='float32')
network = TestModel()
model = tf.keras.Model(inputs, network(inputs))
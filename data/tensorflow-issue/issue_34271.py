from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, sys, time


class Customized_DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, **kwargs):
        self.units = units
        self.activation = tf.keras.layers.Activation(activation)
        super(Customized_DenseLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (input_shape[-1], self.units),
                                      dtype = tf.float32,
                                      initializer = 'uniform',
                                      trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                    shape = (self.units,),
                                    dtype = tf.float32,
                                    initializer = 'zeros',
                                    trainable = True)
        super(Customized_DenseLayer, self).build(input_shape)
    def call(self, inputs):
        return self.activation(inputs @ self.kernel + self.bias)


housing = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state = 11)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

model = tf.keras.models.Sequential()
model.add(Customized_DenseLayer(units = 30, input_shape = (8,), activation = 'relu'))
model.add(Customized_DenseLayer(units = 1))

print(model.summary())

model.compile(optimizer = tf.keras.optimizers.SGD(0.005), loss = 'mean_squared_error')

callbacks = [tf.keras.callbacks.EarlyStopping(min_delta = 0.01, patience = 5, verbose = 1)]

history = model.fit(x_train_scaled, y_train, epochs = 100,
                    validation_data = (x_valid_scaled, y_valid),
                    callbacks = callbacks, verbose = 1)

print(history.history)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

result = model.evaluate(x_test_scaled, y_test)
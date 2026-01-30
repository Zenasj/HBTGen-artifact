import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend

class MyLayer(keras.layers.Layer):
    @tf.function
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        tf.print("training: ", training)
        return keras.backend.in_test_phase(inputs + 1., inputs + 2., training)

X = np.zeros((1, 2))

print("Test A: MyLayer()(X)")
y_pred = MyLayer()(X)
print("Pred:", y_pred)
print("training should be 0\n")

print("Test B: MyLayer()(X, training=True)")
y_pred = MyLayer()(X, training=True)
print("Pred:", y_pred)
print("training should be True\n")

print("Test C: model = ...; model.compile(...); model.fit(...)")
model = keras.models.Sequential([MyLayer(input_shape=[2])])
model.compile(loss="mse", optimizer="sgd")
history = model.fit(X, X, epochs=2)
print("training should be 1 (used to fail, now passes)\n")

print("Test D: K.set_learning_phase(1); MyLayer()(X)")
K.set_learning_phase(1)
y_pred = MyLayer()(X)
print("Pred:", y_pred)
print("training should be 1\n")
K.clear_session()

print("Test E: layer = MyLayer(); K.set_learning_phase(1); layer(X)")
layer = MyLayer()
K.set_learning_phase(1)
y_pred = layer(X)
print("Pred:", y_pred)
print("training should be 1\n")
K.clear_session()

print("Test F: K.set_learning_phase(1); layer = MyLayer(); K.set_learning_phase(0); layer(X)")
K.set_learning_phase(1)
layer = MyLayer()
K.set_learning_phase(0)
y_pred = layer(X)
print("Pred:", y_pred)
print("training should be 0\n")
K.clear_session()

print("Test G: K.set_learning_phase(1); model = ...; model.compile(...); model.fit(...)")
K.set_learning_phase(1)
model = keras.models.Sequential([MyLayer(input_shape=[2])])
model.compile(loss="mse", optimizer="sgd")
history = model.fit(X, X, epochs=2)
print("training should be 1\n")
K.clear_session()

print("Test H: model = ...; model.compile(...); K.set_learning_phase(1); model.fit(...)")
model = keras.models.Sequential([MyLayer(input_shape=[2])])
model.compile(loss="mse", optimizer="sgd")
K.set_learning_phase(1)
history = model.fit(X, X, epochs=2)
print("training should be 1 (ERROR?)\n")
K.clear_session()

print("Test I: K.set_learning_phase(1); model = ...; K.set_learning_phase(0); model.compile(...); model.fit(...)")
K.set_learning_phase(1)
model = keras.models.Sequential([MyLayer(input_shape=[2])])
K.set_learning_phase(0)
model.compile(loss="mse", optimizer="sgd")
history = model.fit(X, X, epochs=2)
print("This test does not make much sense, why would you call fit with learning phase 0?\n")
K.clear_session()

from tensorflow import keras
import numpy as np

model = keras.models.Sequential([keras.layers.Dense(1)])
model.compile(loss="mse", optimizer="sgd")
model.fit(np.random.rand(1000, 1), np.random.rand(1000, 1))

with keras.backend.learning_phase_scope(1):
    model.fit(np.random.rand(1000, 1), np.random.rand(1000, 1))

import numpy as np
import tensorflow as tf
from tensorflow import keras

class MyLayer(keras.layers.Layer):
    def call(self, inputs, training=None):
        print("(tracing) training =", training)
        tf.print("(running) training =", training)
        return inputs + 1.

model = keras.models.Sequential([MyLayer()])
model.compile(loss="mse", optimizer="sgd")
X_train, Y_train, X_valid, Y_valid = np.random.rand(4, 100, 2)
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid))
Y_pred = model.predict(X_valid)

import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend

X = np.zeros((1, 2))

class MyLayer(keras.layers.Layer):
    @tf.function
    def call(self, inputs, training=None):
        tf.print("training: ", training)
        tf.print("K.learning_phase(): ", K.learning_phase())
        return keras.backend.in_test_phase(inputs + 1., inputs + 2., training)

model = keras.models.Sequential([MyLayer(input_shape=[2])])
model.compile(loss="mse", optimizer="sgd")
print("_" * 80)
print(">>> model.fit(...)")
history = model.fit(X, X, epochs=2)
print("_" * 80)
print(">>> model(X)")
print("=>", model(X))
print("_" * 80)
print(">>> model(X, training=True)")
print("=>", model(X, training=True))
print("_" * 80)
print(">>> model(X, training=False)")
print("=>", model(X, training=False))
print("_" * 80)
print(">>> with K.learning_phase_scope(1): model(X)")
with K.learning_phase_scope(1):
    print("=>", model(X))
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

# fake data
X = np.random.rand(100, 10)
Y = np.random.rand(100, 5)
r = np.random.rand(5)

# build/compile/fit model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation="relu", name="layer1"),
        tf.keras.layers.Dense(10, activation="relu", name="layer2"),
        tf.keras.layers.Dense(5, name="layer3"),
    ]
)
model.compile(optimizer="adam", loss="mse")
model.fit(X, Y, epochs=50)

# add rescaling layer
model.add(tf.keras.layers.Rescaling(r))

# test point
x_tst = np.random.rand(1, 10)

# this works!
print(model(x_tst))

# save model
model.save('model.keras')

# load model now
model = tf.keras.models.load_model('model.keras')
model.summary()

# error here!
print(model(x_tst))

import numpy as np
import tensorflow as tf

# Fake data
X = np.random.rand(100, 10)
Y = np.random.rand(100, 5)
r = np.random.rand(5)

# Build/compile/fit model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", name="layer1"),
    tf.keras.layers.Dense(10, activation="relu", name="layer2"),
    tf.keras.layers.Dense(5, name="layer3"),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, Y, epochs=50)

# Save model
model.save('model.keras')

# Load model
model = tf.keras.models.load_model('model.keras')

# Add rescaling layer after loading
model.add(tf.keras.layers.Rescaling(r))

# Test point
x_tst = np.random.rand(1, 10)

# Print prediction (should work after recompiling)
print(model(x_tst))

# Summary of the model
model.summary()

# Recompile the model after adding Rescaling layer
model.compile(optimizer="adam", loss="mse")

# Now the prediction would run successful
print(model(x_tst))

# fake data
X = np.random.rand(100, 10)
Y = np.random.rand(100, 5)
r = np.random.rand(5)

# build/compile/fit model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation="relu", name="layer1"),
        tf.keras.layers.Dense(10, activation="relu", name="layer2"),
        tf.keras.layers.Dense(5, name="layer3"),
    ]
)
model.compile(optimizer="adam", loss="mse")
model.fit(X, Y, epochs=50)

# add rescaling layer
model.add(tf.keras.layers.Rescaling(r))

# test point
x_tst = np.random.rand(1, 10)

# this works!
print(model(x_tst))

# save model
model.save('model.keras')

# load model now
model = tf.keras.models.load_model('model.keras')

# Recompile the model after adding Rescaling layer
model.compile(optimizer="adam", loss="mse")

# error here!
print(model(x_tst))
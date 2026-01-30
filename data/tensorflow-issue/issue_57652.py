from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

"tf.compat.v1.disable_eager_execution()"

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

from datetime import datetime

import streamlit as st

# Do not use GPU
tf.config.set_visible_devices([], "GPU")
tf.compat.v1.disable_eager_execution()  # adding this line make keras training much slower

# Generate data
def f(x):
    return (x + 1) * np.sin(5 * x)


x_plot = np.arange(-1, 1 + 0.001, 0.001)
y_plot = f(x_plot)

x_train = np.arange(-1 + 0.05, 1, 0.2)
y_train = f(x_train)

x_val = np.arange(-1 + 0.15, 1, 0.2)
y_val = f(x_val)

# Plot the problem
plt.figure()
plt.plot(x_plot, y_plot, "-", label="Orgininal function")
plt.plot(x_train, y_train, "o", label="Training points")
plt.plot(x_val, y_val, "s", label="Validation points")
plt.xlim(-1, 1)
plt.ylim(-2, 2)
plt.xlabel("x")
plt.ylabel("f")
plt.grid()
plt.legend()
plt.show(block=False)

# Reshape
X_train = x_train.reshape(x_train.shape[0], 1)
Y_train = y_train.reshape(x_train.shape[0], 1)

X_val = x_val.reshape(x_val.shape[0], 1)
Y_val = y_val.reshape(x_val.shape[0], 1)

# Simple model
start_time = datetime.now()


@st.experimental_singleton
def train():
    tf.keras.utils.set_random_seed(1)
    model = Sequential()
    model.add(Input(shape=(1,)))  # Input layer
    model.add(
        Dense(
            4,
            activation="sigmoid",
            kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=1),
        )
    )
    model.add(Dense(1))

    model.compile(
        loss="mean_squared_error", optimizer=Adam(learning_rate=3e-1), run_eagerly=False
    )
    history = model.fit(
        X_train,
        Y_train,
        validation_split=0.0,
        validation_data=(X_val, Y_val),
        validation_freq=1,
        batch_size=X_train.shape[0],
        epochs=2000,
        verbose=0,
    )
    return model


run_time = datetime.now() - start_time

model2 = train()

print("Training time : {:.4f} s".format(run_time.total_seconds()))

st.write("Please rerun the app to see the error")

model2.predict(X_val)
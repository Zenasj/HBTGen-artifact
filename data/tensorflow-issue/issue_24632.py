import random
from tensorflow.keras import layers
from tensorflow.keras import models

import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

X_train = np.random.rand(1000, 10)
y_train = np.random.rand(1000)
model = keras.models.Sequential([keras.layers.Dense(1)])
model.compile(loss="mse", optimizer="sgd")
tensorboard_cb = keras.callbacks.TensorBoard("logs/run1")
model.fit(X_train, y_train, epochs=1000, callbacks=[tensorboard_cb])
# NOTE: you must interrupt training (Ctrl-C) before it finishes

# For issue #1, try this:
model.fit(X_train, y_train, epochs=1000, callbacks=[tensorboard_cb])

# For issue #2, try this (you may need to interrupt and retry a few times):
shutil.rmtree("logs")
model = keras.models.Sequential([keras.layers.Dense(1)])
model.compile(loss="mse", optimizer="sgd")
tensorboard_cb = keras.callbacks.TensorBoard("logs/run1")
model.fit(X_train, y_train, epochs=1000, callbacks=[tensorboard_cb])
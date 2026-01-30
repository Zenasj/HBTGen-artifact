import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
from tensorflow import keras

X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

model = keras.models.Sequential([
    keras.layers.Dense(2, activation="relu", input_shape=[10]),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="sgd", metrics=["mae", "mse"])
model.fit(X_train, y_train, epochs=2)

assert len(model.metrics) == 2
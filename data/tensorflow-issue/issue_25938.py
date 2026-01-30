import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

X_train = np.random.randn(100, 2)
y_train = np.random.randn(100, 1)

model = keras.models.Sequential([keras.layers.Dense(1, input_dim=2)])
model.compile(loss=keras.losses.Huber(2.0), optimizer="sgd")
model.fit(X_train, y_train, epochs=2)
model.save("my_model.h5")
model = keras.models.load_model("my_model.h5") # Raises a ValueErro

from tensorflow.python.keras import losses

model = load_model("test/test_model_ep09.h5", {"TripletLoss": TripletLoss, "PlainBlock": PlainBlock})
model.predict(x_val)
import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

X_train = np.random.rand(100, 3)
y_train = np.random.rand(100, 1)
model = keras.models.Sequential([keras.layers.Dense(1)])
model.compile(loss="mse", optimizer="sgd")
model.fit(X_train, y_train)

model_version = 1
model_path = os.path.join("my_model", str(model_version))
os.makedirs(model_path)
tf.saved_model.save(model, model_path)
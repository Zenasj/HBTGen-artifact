import random
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.Model(...)
tf.saved_model.save(model, path)
imported = tf.saved_model.load(path)
outputs = imported(inputs)

import tensorflow as tf
from tensorflow import keras
import numpy as np

path = "my_saved_model"
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100, 1)

model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[5])])
model.compile(loss="mse", optimizer="sgd")
model.fit(X_train, y_train)

tf.saved_model.save(model, path)

imported = tf.saved_model.load(path)

inputs = keras.layers.Input(shape=[5])
outputs = imported(inputs) # Raises _SymbolicException (see stacktrace below) <<<!!!
model = keras.Model(inputs=[inputs], outputs=[outputs])
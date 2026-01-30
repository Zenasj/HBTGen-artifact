import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=2)
print(model.evaluate(X_test, y_test))
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

reconstructed_model = tf.keras.models.load_model("my_model")

import tensorflow as tf
import numpy as np

def get_model():
    # Create a simple model.
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = tf.keras.models.load_model("my_model")

print(reconstructed_model.optimizer.weights)

reconstructed_model = tf.keras.models.load_model("my_model", compile=False)
for w in reconstructed_model.optimizer.weights:
    print(w.shape)

reconstructed_model.compile(reconstructed_model.optimizer, loss="mean_squared_error")
reconstructed_model.fit(test_input, test_target)
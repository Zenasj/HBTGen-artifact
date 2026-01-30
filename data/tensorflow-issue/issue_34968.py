from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
X = [0.42]
Y = [[0, 1]]
model.fit(X, Y, epochs=1)
pred_before = model.predict(X)
print(f"*** pred : {pred_before} ***")
model.save("./mymodel/1", save_format="tf")
model = tf.keras.models.load_model("./mymodel/1")
model.predict([X])
print(f"*** signatures : {model.signatures} ***")
pred_after = model([X])
np.testing.assert_almost_equal(pred_before, pred_after)
print(model.predict([X]))
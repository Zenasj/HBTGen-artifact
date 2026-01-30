from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import layers


inputs = tf.keras.Input(shape=(100, 100, 3))
normalized = layers.Lambda(tf.image.per_image_standardization)(inputs)
dense1 = layers.Dense(100, activation="relu")(normalized)
dense2 = layers.Dense(10, activation="softmax")(dense1)
model = tf.keras.Model(inputs=inputs, outputs=dense2)

model.save("my_model.hd5", save_format="h5")

new_model = tf.keras.models.load_model("my_model.hd5")  # <-- Error here
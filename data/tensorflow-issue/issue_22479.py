import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.models.load_model(path_model)
model.pop()  # to remove last layer
new_model = tf.keras.Sequential([
	model,
	tf.keras.layers.Dense(units=nb_classes, activation="softmax", name="new_layer_name")
])
new_model.build((None, height, width, 3))
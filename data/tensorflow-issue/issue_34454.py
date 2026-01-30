from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

shape = (224, 224, 3)

# functional model
base_model2 = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
inputs = tf.keras.Input(shape=shape, name="input")
x = base_model2(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu", name="embeddings")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)


tf.keras.models.save_model(model2, "model")
model_l2 = tf.keras.models.load_model("model")

# this raises exception
model_loaded = tf.keras.Model(
    inputs=model_l2.input, outputs=[model_l2.get_layer(layer_name).output for layer_name in ["probs", "embeddings"]]
)

model3.add(keras.layers.Dense(100,input_shape=(500,)))
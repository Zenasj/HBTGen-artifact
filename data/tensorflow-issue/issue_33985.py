from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class L2NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, name="normalize", **kwargs):
        super(L2NormalizeLayer, self).__init__(name=name, **kwargs)

    def call(self, input):
        return tf.keras.backend.l2_normalize(input, axis=1)

    def get_config(self):
        config = super(L2NormalizeLayer, self).get_config()
        return config

shape = (224, 224, 3)

# functional model
base_model2 = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
inputs = tf.keras.Input(shape=shape, name="input")
x = base_model2(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
y = L2NormalizeLayer(name="embeddings")(x)
#y = tf.keras.layers.Lambda(lambda k: tf.keras.backend.l2_normalize(k, axis=1), name="embeddings")(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

# after training model i would like to load it and extract probs with embeddings
tf.keras.models.save_model(model2, "model.h5")
model_l2 = tf.keras.models.load_model("model.h5")

model_loaded = tf.keras.Model(
    inputs=model_l2.input, outputs=[model_l2.get_layer(layer_name).output for layer_name in ["probs", "embeddings"]]
)
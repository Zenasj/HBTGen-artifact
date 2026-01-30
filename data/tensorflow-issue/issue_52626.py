from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

if __name__ == "__main__":
    input_layer = tf.keras.Input(shape=[100])
    dense_layer = tf.keras.layers.Dense(1)(input_layer)
    concatenate_layer = tf.keras.layers.Concatenate()([dense_layer])
    model = tf.keras.Model([input_layer], [concatenate_layer])
    model.compile(optimizer="adam", loss="mean_absolute_error")
    model.save("model.h5")
    loaded_model = tf.keras.models.load_model("model.h5")
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def unfused_model():
    img = tf.keras.layers.Input(shape=(3, 3, 10))
    x = tf.keras.layers.Conv2D(100, (3, 3))(img)
    x = tf.keras.layers.Reshape((50, 2))(x)
    return tf.keras.Model(img, x)

def fused_model():
    img = tf.keras.layers.Input(shape=(3, 3, 10))
    x = tf.keras.layers.Conv2D(100, (3, 3))(img)
    x = tf.reshape(x, (-1, 50, 2))
    return tf.keras.Model(img, x)

def convert_model(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(filename, "wb") as f:
        f.write(converter.convert())

convert_model(unfused_model(), "/tmp/unfused_model.tflite")
convert_model(fused_model(), "/tmp/fused_model.tflite")
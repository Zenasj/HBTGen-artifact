import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pathlib

inpt = tf.keras.layers.Input(shape=[256, 256, 3])
out = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x))(inpt)
out = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x))(out)
model = tf.keras.Model(inpt, out)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
pathlib.Path('out.tflite').write_bytes(tflite_model)
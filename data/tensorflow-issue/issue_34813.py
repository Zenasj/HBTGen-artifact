import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_tensor = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.GRU(154)(input_tensor)
x = tf.keras.layers.Flatten()(x)
out_tensor = tf.keras.layers.Dense(10, activation="relu")(x)
model = tf.keras.Model(input_tensor, out_tensor)
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert() # where is wrong
open("converted_keras_model.tflite", "wb").write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
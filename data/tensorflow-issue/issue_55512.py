from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf

def create_model():
    inputs = tf.keras.Input(shape=[5, 5, 3])
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    model = tf.keras.Model(inputs, x)
    return model


model = create_model()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model1 = converter.convert()

tflite_model2 = converter.convert()

with open("model1.tflite", "wb") as f:
    f.write(tflite_model1)

with open("model2.tflite", "wb") as f:
    f.write(tflite_model2)

print(os.system("diff model1.tflite model2.tflite"))


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model3 = converter.convert()

with open("model3.tflite", "wb") as f:
    f.write(tflite_model3)

print(os.system("diff model1.tflite model3.tflite"))
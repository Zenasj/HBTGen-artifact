from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

input_layer = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float16)
final = tf.keras.layers.Dense(10, dtype=tf.float32)(input_layer)
model = tf.keras.models.Model(input_layer, final)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

import tensorflow as tf

input_layer = tf.keras.layers.Input(shape=(224,224,3))
final = tf.keras.layers.Dense(10, dtype=tf.float16)(input_layer)
model = tf.keras.models.Model(input_layer, final)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

converter.experimental_new_converter = True
converter.target_spec.supported_ops =[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
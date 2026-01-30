from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf

model_dir = 'model'

converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()
fo = open("model.tflite", "wb")
fo.write(tflite_model)
fo.close

model_dir = 'model'
saved_model = tf.saved_model.load(export_dir=model_dir)
Signature =  saved_model.signatures
print(Signature, "Len: ", len(Signature))

import tensorflow as tf
model_dir = 'model'
h5_model = "mymodel.h5"
model = tf.saved_model.load(model_dir)
# model = tf.keras.models.load_model(model_dir) # trying both method to load model
tf.keras.models.save_model(model, h5_model)

import tensorflow_hub as hub
model_dir = 'model'
model_layers = hub.KerasLayer(model_dir)

model = tf.saved_model.load(model_dir)

keras_model = tf.keras.Sequential()
keras_model.add(...)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
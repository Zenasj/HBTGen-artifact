from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32,32)),
    tf.keras.layers.ELU()
])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter=False
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open('elu.tflite', "wb") as f:
    f.write(tflite_model)
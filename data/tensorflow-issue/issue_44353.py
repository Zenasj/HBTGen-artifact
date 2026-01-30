from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def mymodel():
    img = tf.keras.layers.Input(shape=(96, 96, 3))
    x = img
    x = tf.quantization.fake_quant_with_min_max_vars(x, -3, 3)
    x = tf.keras.layers.Conv2D(32, 3)(x)
    x = tf.quantization.fake_quant_with_min_max_vars(x, -3, 3)
    return tf.keras.Model(img, x)

converter = tf.lite.TFLiteConverter.from_keras_model(mymodel())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
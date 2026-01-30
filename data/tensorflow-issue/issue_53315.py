from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

input = tf.keras.Input(shape=[1, 3, 8, 8, 8])
output = tf.keras.layers.Conv3D(
    filters=12,
    kernel_size=(2, 2, 2),
    data_format="channels_first",
    use_bias=False,
)(input)
model = tf.keras.Model(input, output)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(
            input_shape=(2, 4, 4, 3), batch_size=1
        ),
        tf.keras.layers.Conv3D(filters=3, kernel_size=(2, 4, 4), groups=3),
    ]
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_quant_model = converter.convert()
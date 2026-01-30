import random

import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential

_layers = [
    layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(10),
]
mock_model = Sequential(_layers)

def convert_to_fp16(tf_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def quantize_model(tf_model, input_shape):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Float fallback for operators that do not have an integer implementation
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape)
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


assert sys.getsizeof(convert_to_fp16(mock_model)) > sys.getsizeof(quantize_model(mock_model, (224, 224, 3)))
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os
tf.get_logger().setLevel('ERROR')
import tensorboard as tb
import numpy as np
import librosa
import matplotlib.pyplot as plt
import datetime

def converter_issue():

    input_shape = (1,)

    def representative_ds():
        for _ in range(100):
            x = np.random.uniform(0, 2*np.pi, size=input_shape)
            yield [x.astype(np.float32)]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Dense(16),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(16),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1)
    ])

    model.summary()

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_type = tf.int8
    converter.inference_input_type = tf.int8  
    converter.inference_output_type = tf.int8 

    tflite_quant_model = converter.convert()

    return tflite_quant_model


if __name__ == "__main__":
    model = converter_issue()
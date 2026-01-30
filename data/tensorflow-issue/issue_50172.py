import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


def representative_dataset():
    for _ in range(10):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float32)]


dil = 2
for ptq in [False, True]:
    _in = tf.keras.Input((244, 244, 3))
    if ptq:
        x = _in
    else:
        x = tf.quantization.fake_quant_with_min_max_args(_in)

    x = tf.keras.layers.Conv2D(16, 3, padding='same', dilation_rate=(dil, dil), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if not ptq:
        x = tf.quantization.fake_quant_with_min_max_args(x)
    model = tf.keras.Model(_in, x)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if ptq:
        converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()

    s = '_ptq' if ptq else ''
    tflite_file = f'/tmp/dilated_conv2d/dil{dil}{s}{c}.tflite'
    with open(tflite_file, 'wb') as f:
        f.write(tflite_quant_model)

    print(f'Generated {tflite_file} with TF {tf.__version__}')
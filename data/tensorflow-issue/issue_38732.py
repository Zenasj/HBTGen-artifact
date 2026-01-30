from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import pathlib
import numpy

def convert(model, name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    pathlib.Path(name).write_bytes(tflite_model)
    

inpt = tf.keras.layers.Input(shape=[256, 256, 3])
conv = tf.keras.layers.Conv2D(32, 1, padding="same")(inpt)

out = tf.keras.layers.Lambda(
    lambda x: tf.nn.max_pool2d(
        x, 16, strides=1, padding="SAME"
    )
)(conv)
model = tf.keras.Model(inpt, [conv, out])
convert(model, 'out_gpu.tflite')

# Now force this to the maxpool to run on the cpu
inpt = tf.keras.layers.Input(shape=[256, 256, 3])
conv = tf.keras.layers.Conv2D(32, 1, padding="same")(inpt)
# This op should force the tensor onto the cpu.
out = tf.where(tf.equal(conv, conv), conv, tf.zeros_like(conv))
out = tf.keras.layers.Lambda(
    lambda x: tf.nn.max_pool2d(
        x, 16, strides=1, padding="SAME"
    )
)(out)
model = tf.keras.Model(inpt, [conv, out])
convert(model, 'out_cpu.tflite')
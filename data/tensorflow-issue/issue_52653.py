from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from pathlib import Path
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224,224,3), alpha=1.0, weights='imagenet',
        classes=6, include_top=False, pooling='avg')
  output = tf.keras.layers.Dense(6, name='logits', dtype='float32')(base_model.output)
  model = tf.keras.Model(base_model.inputs, output)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
Path('model.fp32.tflite').write_bytes(tflite_model)
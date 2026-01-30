import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

img = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(16, 3)(img)
x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
model = tf.keras.Model(img, x)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

model = tf.keras.models.clone_model(
    model,
    clone_function=lambda layer: layer.__class__.from_config(
        {**layer.get_config(), "dtype": "float32"}
    ),
)

import tensorflow as tf
import numpy as np

# Build a toy model
def get_model():
  img = tf.keras.layers.Input(shape=(224, 224, 3))
  x = tf.keras.layers.Conv2D(16, 3)(img)
  x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
  model = tf.keras.Model(img, x)
  return model

# Create a mixed_float16 model
tf.keras.mixed_precision.set_global_policy("mixed_float16")
model = get_model()

# Compile and train model
model.compile('sgd', 'mse')
x = np.random.normal(size=(64, 224, 224, 3))
y = np.random.normal(size=(64, 222,222, 16))
model.fit(x, y)

# Create a float32 model with the same weights as the mixed_float16 model, so
# that it loads into TF Lite
tf.keras.mixed_precision.set_global_policy("float32")
f32_model = get_model()
f32_model.set_weights(model.get_weights())

# Load model into TF lite
converter = tf.lite.TFLiteConverter.from_keras_model(f32_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
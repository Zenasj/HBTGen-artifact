from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

x_inputs = tf.keras.layers.Input((3,229,1))
x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(x_inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.MaxPooling2D((1,2))(x)
x = tf.keras.layers.Conv2D(64, (3,3), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((1,2))(x)
x = tf.keras.layers.Reshape((-1,tf.keras.backend.int_shape(x)[-1] * tf.keras.backend.int_shape(x)[-2]))(x)
x = tf.keras.layers.LSTM(256)(x)
x = tf.keras.layers.Dense(1)(x)
x = tf.keras.layers.Activation('sigmoid')(x)
model = tf.keras.models.Model(x_inputs, x)
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
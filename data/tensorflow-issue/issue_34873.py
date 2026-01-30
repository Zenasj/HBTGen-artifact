import random
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.models.load_model('my_conv.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open("custom_cnn_f16.tflite", "wb").write(tflite_model)

import tensorflow as tf # 2.0
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D, Dropout
import numpy as np

data = np.random.uniform(0, 1, (1000, 100, 100, 1)).astype(np.float32)
labels = np.random.randint(0, 9, (1000,)).astype(np.int32)
dataset = tf.data.Dataset.from_tensor_slices((data, labels))

layers = []
layers.append(Conv2D(64, 3, padding='same', activation='relu', name='conv_1', input_shape=(100, 100, 1)))
layers.append(MaxPooling2D(2, name='max_pool_1'))
layers.append(Dropout(0.2, name='dropout_1'))
layers.append(Conv2D(128, 3, padding='same', activation='relu', name='conv_2'))
layers.append(MaxPooling2D(2, name='max_pool_2'))
layers.append(Dropout(0.2, name='dropout_2'))
layers.append(Conv2D(256, 3, padding='same', activation='relu', name='conv_3'))
layers.append(MaxPooling2D(2, name='max_pool_3'))
layers.append(Dropout(0.2, name='dropout_3'))
layers.append(Flatten(name='flatten'))
layers.append(Dense(128, activation='relu', name='dense_1'))
layers.append(Dense(64, activation='relu', name='dense_2'))
layers.append(Dense(10, activation='softmax', name='dense_3'))
model = keras.Sequential(layers, name='my_cnn')
model.summary()

dataset = dataset.batch(8)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=1)

# Convert to TFLite with Float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open("my_f16_cnn.tflite", "wb").write(tflite_model)